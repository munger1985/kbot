(function () {
  "use strict";
  const appApi = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let sourceReloadTimer = null;
  let sourceReloadAttempts = 0;
  const configs = {
    targets: { path: "/targets", cols: [["display_name", "目标"], ["importance_level", "重要程度", "importance"], ["db_type", "数据库"], ["_access", "访问模式", "target-access"], ["status", "启用状态", "badge"], ["connectivity_status", "连通性", "badge"], ["observed_status", "观测状态", "badge"], ["updated_at", "更新时间", "date"], ["_actions", "操作", "target-actions"]] },
    "diagnostic-sources": { path: "/diagnostic-sources", cols: [["display_name", "监控源"], ["source_type", "类型"], ["status", "启用状态", "badge"], ["connectivity_status", "连通性", "badge"], ["updated_at", "更新时间", "date"], ["_actions", "操作", "source-actions"]], detail: "diagnostic-source-detail.html?id=" },
    "inspection-plans": { path: "/inspection-plans", cols: [["display_name", "计划"], ["agent_name", "DBA Agent"], ["schedule_type", "调度周期", "schedule"], ["status", "状态", "badge"], ["_latest_fire", "最近执行", "latest-inspection"], ["next_run_at", "下次执行", "date"], ["_actions", "操作", "inspection-actions"]] },
    reports: { path: "/reports", render: "report-list", detail: "report-detail.html?id=" },
    runs: { path: "/runs?limit=100", cols: [["workflow_kind", "工作流", "workflow"], ["trigger_type", "触发方式", "trigger"], ["status", "状态", "run-status"], ["created_at", "创建时间", "date"], ["completed_at", "完成时间", "date"], ["ops_run_id", "运行 ID", "id"]], detail: "run-detail.html?id=" },
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
      const lifecycle = item.status === "ACTIVE"
        ? '<button type="button" data-inspection-action="pause">暂停</button><button type="button" data-inspection-action="disable">停用</button>'
        : item.status === "PAUSED"
          ? '<button type="button" class="primary" data-inspection-action="activate">启用</button><button type="button" data-inspection-action="disable">停用</button>'
          : item.status === "DISABLED"
            ? '<button type="button" class="primary" data-inspection-action="activate">启用</button><button type="button" class="danger" data-inspection-action="delete">删除</button>'
            : "";
      const report = item._latest_fire ? '<button type="button" data-inspection-action="reports">查看报告</button>' : "";
      return `<div class="ops-actions">${report}<button type="button" data-inspection-action="detail">详情</button><button type="button" data-inspection-action="edit">编辑</button>${lifecycle}</div>`;
    }
    if (type === "latest-inspection") {
      const latest = item._latest_fire;
      return latest
        ? `${shell.badge(latest.status)}<small>${shell.escape(latest.error_message || shell.fmt(latest.scheduled_at))}</small>`
        : '<span class="ops-badge">尚未执行</span>';
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
    if (type === "workflow") {
      return shell.escape({ CHAT_TURN: "智能诊断", ALERT_DIAGNOSIS: "告警诊断", INSPECTION: "日常巡检", CHANGE: "受控变更", VERIFICATION: "结果验证" }[value] || value || "—");
    }
    if (type === "trigger") {
      return shell.escape({ CHAT: "对话发起", API: "接口触发", ROOT: "根任务", ALERT: "监控告警", SCHEDULE: "巡检计划" }[value] || value || "—");
    }
    if (type === "run-status") {
      const labels = { CREATED: "已创建", QUEUED: "等待执行", RUNNING: "运行中", WAITING_INPUT: "等待补充信息", WAITING_APPROVAL: "等待审批", COMPLETED: "已完成", PARTIAL: "部分完成", FAILED: "失败", CANCELLED: "已取消", EXPIRED: "已过期" };
      return shell.badge(value, labels[value] || value || "未知");
    }
    if (type === "importance") {
      const labels = { 1: "低", 2: "一般", 3: "重要", 4: "关键", 5: "核心" };
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
    return `<a class="ops-report-row" href="${detail}${encodeURIComponent(item.report_id)}"><div class="ops-report-primary"><div class="ops-report-kicker"><span>${shell.escape(reportTypeLabel(item.report_type))}</span><span>报告 v${shell.escape(item.report_version)}</span><span>${shell.escape(item.template_ref || "系统模板")} · 模板 v${shell.escape(item.template_version || "1")}</span></div><strong>${shell.escape(item.title || "正式报告")}</strong><p>${shell.escape(reportDisplayText(item.summary || "报告未提供摘要"))}</p></div><div class="ops-report-meta"><small>报告生成时间</small><time>${shell.escape(shell.fmt(publishedAt))}</time></div><div class="ops-report-state">${reportStatusBadge(item.status)}<span>查看报告</span></div></a>`;
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
      const [payload, agentRows, firePage] = await Promise.all([
        KBotAIOpsAuth.request(appApi + cfg.path),
        page === "inspection-plans" ? KBotAIOpsAuth.request(`${appApi}/agents`) : Promise.resolve([]),
        page === "inspection-plans" ? KBotAIOpsAuth.request(`${appApi}/inspection-fires?limit=100`) : Promise.resolve({ items: [] }),
      ]);
      const agentNames = new Map((Array.isArray(agentRows) ? agentRows : []).map((agent) => [String(agent.agent_id), agent.display_name]));
      const latestFires = new Map();
      for (const fire of firePage?.items || []) {
        if (!latestFires.has(String(fire.plan_id))) latestFires.set(String(fire.plan_id), fire);
      }
      const sourceItems = Array.isArray(payload) ? payload : payload?.items || [];
      const items = page === "inspection-plans"
        ? sourceItems.map((item) => ({ ...item, agent_name: agentNames.get(String(item.agent_id)) || shell.short(item.agent_id), _latest_fire: latestFires.get(String(item.plan_id)) || null }))
        : sourceItems;
      if (!items.length) {
        body.innerHTML = `<tr><td class="ops-empty" colspan="${cfg.cols.length}">当前范围内暂无数据</td></tr>`;
        return;
      }
      body.innerHTML = items.map((item) => `<tr data-resource-id="${shell.escape(resourceId(item))}"${cfg.detail ? ` data-href="${cfg.detail}${encodeURIComponent(resourceId(item))}"` : ""}>${cfg.cols.map((col) => `<td>${cell(item, col)}</td>`).join("")}</tr>`).join("");
      body.querySelectorAll("[data-href]").forEach((row) => {
        row.style.cursor = "pointer";
        row.addEventListener("click", () => {
          const editors = {
            targets: globalThis.KBotAIOpsTargets,
          };
          if (editors[page]?.openEdit) {
            editors[page].openEdit(row.dataset.resourceId);
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
            const action = button.dataset.inspectionAction;
            if (action === "reports") {
              location.href = `inspections.html?plan_id=${encodeURIComponent(item.plan_id)}`;
              return;
            }
            if (action === "detail") {
              location.href = `inspection-plan-detail.html?id=${encodeURIComponent(item.plan_id)}`;
              return;
            }
            if (action === "edit") {
              await globalThis.KBotAIOpsConfigurations?.openEdit?.("inspection-plans", item.plan_id);
              return;
            }
            if (action === "delete" && !globalThis.confirm(`确认永久删除巡检计划“${item.display_name}”？只有未产生执行记录的已停用计划可以删除。`)) return;
            button.disabled = true;
            try {
              const deleting = action === "delete";
              await KBotAIOpsAuth.request(`${appApi}/inspection-plans/${encodeURIComponent(item.plan_id)}${deleting ? "" : `/${action}`}`, {
                method: deleting ? "DELETE" : "POST",
                headers: {
                  "If-Match": `"rv-${item.row_version}"`,
                  "Idempotency-Key": KBotAIOpsAuth.uuid(),
                },
                ...(deleting ? {} : { body: JSON.stringify({}) }),
              });
              const messages = {
                activate: "巡检计划已启用",
                pause: "巡检计划已暂停",
                disable: "巡检计划已停用",
                delete: "巡检计划已删除",
              };
              shell.toast(messages[action] || "巡检计划已更新");
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
    const editable = Boolean(document.getElementById("target-facts-form"));
    list.innerHTML = `<table class="ops-table"><thead><tr><th>类型</th><th>事实值</th><th>来源</th><th>确认时间</th>${editable ? "<th>操作</th>" : ""}</tr></thead><tbody>${items.map((item) => `<tr><td>${shell.escape(factTypeLabels[item.fact_type] || item.fact_type)}</td><td>${shell.escape(factValueText(item))}</td><td>${shell.escape(item.source || "—")}</td><td>${shell.escape(shell.fmt(item.confirmed_at))}</td>${editable ? `<td><button type="button" data-retire-target-fact="${shell.escape(item.fact_id)}" data-version="${shell.escape(item.row_version)}">撤回</button></td>` : ""}</tr>`).join("")}</tbody></table>`;
  }

  async function loadTargetFacts(targetId) {
    const payload = await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/facts`);
    renderTargetFacts(payload.items || []);
  }

  async function initializeTargetFacts(targetId) {
    const form = document.getElementById("target-facts-form");
    const list = document.getElementById("target-facts-list");
    const result = document.getElementById("target-facts-result");
    if (!list) return;
    if (!form) {
      try { await loadTargetFacts(targetId); }
      catch (error) { list.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`; }
      return;
    }
    form.dataset.targetId = targetId;
    if (!result) return;
    if (form.dataset.initialized !== "true") {
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
        const currentTargetId = form.dataset.targetId;
        await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(currentTargetId)}/facts`, {
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
        await loadTargetFacts(currentTargetId);
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
        const currentTargetId = form.dataset.targetId;
        await KBotAIOpsAuth.request(
          `${appApi}/targets/${encodeURIComponent(currentTargetId)}/facts/${encodeURIComponent(button.dataset.retireTargetFact)}/retire`,
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
        await loadTargetFacts(currentTargetId);
      } catch (error) {
        result.textContent = error.message;
        result.dataset.tone = "bad";
        button.disabled = false;
      }
      });
      form.dataset.initialized = "true";
    }
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
    const importance = { 1: "低", 2: "一般", 3: "重要", 4: "关键", 5: "核心" }[target.importance_level] || "未知";
    const oracleScope = [
      target.observed_oracle_database_name || target.oracle_pdb_name,
      target.observed_oracle_container_scope || target.oracle_container_scope,
      target.observed_oracle_container_name,
    ].filter(Boolean).join(" · ");
    const row = (label, value) => `<dt>${shell.escape(label)}</dt><dd>${shell.escape(value ?? "—")}</dd>`;
    return `<div class="ops-panel-head target-overview-head"><div><p>${shell.escape(target.db_type)}${target.version_code ? ` · ${shell.escape(target.version_code)}` : ""}</p><h2>${shell.escape(target.display_name)}</h2><small>${shell.escape(environment)} · ${shell.escape(role)} · L${shell.escape(target.importance_level)} ${shell.escape(importance)}</small></div><div class="ops-actions">${shell.badge(target.status)}${shell.badge(target.connectivity_check_pending ? "CHECKING" : target.connectivity_status)}${shell.badge(target.observed_status)}</div></div><div class="ops-panel-body target-overview-grid"><section><h3>数据库范围</h3><dl class="ops-detail">${row("数据库类型", target.db_type)}${row("已验证版本", target.version_code)}${row("环境", environment)}${row("数据库角色", role)}${target.db_type === "ORACLE" ? row("Oracle 范围", oracleScope || "尚未确认") : ""}${row("重要程度", `L${target.importance_level} · ${importance}`)}</dl></section><section><h3>连接状态</h3><dl class="ops-detail">${row("诊断模式", target.readonly_connection_enabled ? "数据库只读直连" : "仅监控数据")}${row("Endpoint", endpoint)}${row("TLS", target.endpoint ? (target.endpoint.tls_enabled ? "已启用" : "未启用") : "不适用")}${row("最近检查", shell.fmt(target.last_connectivity_check_at))}${row("最近成功", shell.fmt(target.last_connectivity_success_at))}${row("最近错误", target.last_error_code || "无")}</dl></section><section><h3>凭据与变更</h3><dl class="ops-detail">${row("诊断凭据", credential(target.diagnostic_credential))}${row("受控变更", target.controlled_change_enabled ? "允许人工审批后执行" : "未启用")}${row("执行凭据", target.controlled_change_enabled ? credential(target.execution_credential) : "不适用")}</dl></section><section><h3>观测与采集</h3><dl class="ops-detail">${row("观测状态", target.observed_status)}${row("最近观测", shell.fmt(target.last_observed_at))}${row("负载快照", target.workload_last_collected_at ? `最近采集 ${shell.fmt(target.workload_last_collected_at)}` : "尚无采集记录")}${row("快照连续失败", target.workload_consecutive_failures)}${row("活动采样", target.activity_sampler_status)}${row("最近采样", shell.fmt(target.activity_last_sampled_at))}${row("采样连续失败", target.activity_consecutive_failures)}</dl></section><section><h3>管理信息</h3><dl class="ops-detail">${row("Target ID", target.target_id)}${row("创建时间", shell.fmt(target.created_at))}${row("创建人", target.created_by)}${row("最后更新", shell.fmt(target.updated_at))}${row("更新人", target.updated_by)}</dl></section></div>`;
  }

  function reportMetadataHtml(data, report, target) {
    const scope = data?.report?.scope || {};
    const healthStatus = String(scope.health_status || "UNKNOWN");
    const healthStatusLabel = {
      NORMAL: "正常",
      INFO: "提示",
      LOW: "低风险",
      MEDIUM: "中风险",
      HIGH: "高风险",
      CRITICAL: "严重风险",
      UNKNOWN: "无法判定",
    }[healthStatus] || healthStatus;
    const rawCoverage = scope.evidence_coverage_percent;
    const hasCoverage = typeof rawCoverage === "number" && Number.isFinite(rawCoverage);
    const coverage = hasCoverage
      ? `${rawCoverage}%${Number.isFinite(Number(scope.evidence_covered_count)) && Number.isFinite(Number(scope.evidence_total_count)) ? ` · ${Number(scope.evidence_covered_count)}/${Number(scope.evidence_total_count)} 项业务检查` : ""}`
      : "历史报告未记录覆盖率";
    const monitoringCoverage = typeof scope.monitoring_sample_coverage_percent === "number"
      && Number.isFinite(scope.monitoring_sample_coverage_percent)
      ? `${scope.monitoring_sample_coverage_percent}%${Number.isFinite(Number(scope.monitoring_low_coverage_metric_count)) && Number.isFinite(Number(scope.monitoring_metric_count)) ? ` · 低覆盖 ${Number(scope.monitoring_low_coverage_metric_count)}/${Number(scope.monitoring_metric_count)} 项` : ""}`
      : "本报告未使用监控时序或未记录采样覆盖率";
    const profile = scope.monitoring_profile?.display_name
      ? `${scope.monitoring_profile.display_name}${scope.monitoring_profile.version ? ` · ${scope.monitoring_profile.version}` : ""}`
      : "";
    const item = (label, value, className = "") => `<div class="${className}"><dt>${shell.escape(label)}</dt><dd>${value}</dd></div>`;
    return `<section class="ops-report-metadata" aria-label="报告信息"><dl>${item("报告类型", shell.escape(reportTypeLabel(report.report_type)))}${item("健康状态", shell.escape(healthStatusLabel))}${item("证据完整性", reportStatusBadge(report.status))}${item("数据库类型", shell.escape(scope.database_type || target?.db_type || "未记录"))}${item("报告版本", `第 ${shell.escape(report.report_version)} 版`)}${item("发布时间", shell.escape(shell.fmt(report.published_at)))}${item("报告周期", `${shell.escape(shell.fmt(report.period_start))} 至 ${shell.escape(shell.fmt(report.period_end))}`)}${item("数据库检查覆盖率", `<strong>${shell.escape(coverage)}</strong><small>${shell.escape(scope.inspection_coverage || "以证据边界章节为准")}</small>`, "ops-report-coverage")}${item("监控采样覆盖率", `<strong>${shell.escape(monitoringCoverage)}</strong>${profile ? `<small>监控口径：${shell.escape(profile)}</small>` : ""}`, "ops-report-coverage")}${item("数据缺口", `${shell.escape(Number.isFinite(Number(scope.evidence_gap_count)) ? Number(scope.evidence_gap_count) : "未记录")} 项`)}${item("关联目标", shell.escape(target?.display_name || "目标已删除或无权访问"))}${item("报告编号", `<code>${shell.escape(report.report_key)}</code>`)}</dl></section>`;
  }

  function dailyTrendSvg(metric, compact = false) {
    const points = (metric.trend_points || [])
      .map((item) => Number(item.baseline_index))
      .filter((value) => Number.isFinite(value));
    if (points.length < 2) return '<span class="daily-no-baseline">基线不足</span>';
    const low = Math.min(85, ...points);
    const high = Math.max(115, ...points);
    const span = Math.max(1, high - low);
    const width = compact ? 112 : 184;
    const height = compact ? 30 : 68;
    const coordinates = points.map((value, index) => {
      const x = points.length === 1 ? width : index * width / (points.length - 1);
      const y = height - 5 - ((value - low) / span) * (height - 10);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const last = coordinates.split(" ").at(-1).split(",");
    const baselineY = (height - 5 - ((100 - low) / span) * (height - 10)).toFixed(1);
    return `<svg class="daily-trend-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${shell.escape(metric.display_name || "指标")}七日趋势"><line x1="0" y1="${baselineY}" x2="${width}" y2="${baselineY}"/><polyline points="${coordinates}"/><circle cx="${last[0]}" cy="${last[1]}" r="2.8"/></svg>`;
  }

  function dailyDeviationHtml(metric) {
    const percent = Number(metric.deviation_percent);
    const available = metric.baseline_status === "AVAILABLE" && Number.isFinite(percent);
    const bounded = available ? Math.max(-50, Math.min(50, percent)) : 0;
    const start = bounded < 0 ? 50 + bounded : 50;
    const width = Math.abs(bounded);
    const percentLabel = available ? `${percent >= 0 ? "+" : ""}${percent.toFixed(2)}%` : "基线不足";
    return `<div class="daily-deviation-row"><div class="daily-metric-name"><strong>${shell.escape(metric.display_name || "指标")}</strong><small>${shell.escape(metric.dimension_label || "全部范围")}</small></div><div class="daily-deviation-track" aria-label="${shell.escape(percentLabel)}"><i></i>${available ? `<span class="${percent >= 0 ? "positive" : "negative"}" style="left:${start}%;width:${width}%"></span>` : ""}</div><div class="daily-current"><strong>${shell.escape(metric.current_display || "无有效值")}</strong><small>当前值</small></div><div class="daily-delta ${available && Math.abs(percent) >= 20 ? "attention" : ""}"><strong>${shell.escape(percentLabel)}</strong><small>${available ? `七日均值 ${shell.escape(metric.baseline_display)}` : `覆盖率 ${shell.escape(metric.coverage_percent ?? "未记录")}%`}</small></div></div>`;
  }

  function dailyInspectionReportHtml(data, report, versions, target) {
    const daily = data.daily_inspection || {};
    const header = daily.header || {};
    const conclusion = daily.conclusion || {};
    const metrics = Array.isArray(daily.baseline_metrics) ? daily.baseline_metrics : [];
    const capacity = Array.isArray(daily.capacity_rows) ? daily.capacity_rows : [];
    const findings = Array.isArray(daily.findings) ? daily.findings : [];
    const findingNotes = Array.isArray(daily.finding_notes) ? daily.finding_notes : [];
    const recommendations = Array.isArray(daily.recommendations) ? daily.recommendations : [];
    const gaps = Array.isArray(daily.gaps) ? daily.gaps : [];
    const boundary = daily.evidence_boundary || {};
    const canEdit = String(versions?.items?.[0]?.report_id || "") === String(report.report_id);
    const versionItems = (versions?.items || []).map((item) => `<option value="${shell.escape(item.report_id)}" ${String(item.report_id) === String(report.report_id) ? "selected" : ""}>v${shell.escape(item.report_version)} · ${shell.escape(shell.fmt(item.published_at))}</option>`).join("");
    const sourceLink = report.source_conversation_id ? `<a href="./chat.html?conversation=${encodeURIComponent(report.source_conversation_id)}">查看来源会话</a>` : "";
    const actionButtons = `${canEdit ? '<button type="button" data-write-ready data-edit-report>编辑报告</button>' : ""}${canManageOperationsKnowledge ? '<button type="button" data-write-ready data-extract-case>提炼为诊断案例</button>' : ""}<button class="primary" type="button" data-write-ready data-download-report>下载 PDF</button>`;
    const metadata = [
      ["报告时间", shell.fmt(header.report_time)],
      ["观测范围", `${shell.fmt(header.observation_start)} 至 ${shell.fmt(header.observation_end)}`],
      ["巡检目标", header.target_name || target?.display_name || "未记录"],
      ["运行环境", header.environment || "未记录"],
      ["数据库类型 / 版本", `${header.database_type || target?.db_type || "未记录"} ${header.database_version || target?.version_code || ""}`.trim()],
      ["报告版本", `v${header.report_version || report.report_version || 1}`],
      ["报告编号", header.report_number || report.report_key || "未记录"],
      ["责任域", header.responsibility_domain || "数据库运维"],
    ].map(([label, value]) => `<div><dt>${shell.escape(label)}</dt><dd>${shell.escape(value)}</dd></div>`).join("");
    const stat = (label, value, className = "") => `<div class="${className}"><span>${shell.escape(label)}</span><strong>${shell.escape(value)}</strong></div>`;
    const coverage = conclusion.evidence_coverage_percent == null ? "未记录" : `${conclusion.evidence_coverage_percent}%`;
    const monitoringCoverage = conclusion.monitoring_coverage_percent == null ? "未记录" : `${conclusion.monitoring_coverage_percent}%`;
    const trendCards = metrics.slice(0, 6).map((metric) => {
      const delta = metric.deviation_percent == null ? "基线不足" : `${Number(metric.deviation_percent) >= 0 ? "+" : ""}${Number(metric.deviation_percent).toFixed(2)}%`;
      return `<figure class="daily-trend-card"><figcaption><strong>${shell.escape(metric.display_name || "指标")}</strong><span>${shell.escape(delta)}</span></figcaption>${dailyTrendSvg(metric)}</figure>`;
    }).join("") || '<p class="daily-empty">当前报告没有可绘制的七日趋势。</p>';
    const capacityRows = capacity.map((metric) => `<tr><td><strong>${shell.escape(metric.display_name || "容量指标")}</strong><small>${shell.escape(metric.dimension_label || "全部范围")}</small></td><td>${shell.escape(metric.current_display || "无有效值")}</td><td>${shell.escape(metric.baseline_display || "基线不足")}</td><td>${metric.deviation_percent == null ? "基线不足" : `${Number(metric.deviation_percent) >= 0 ? "+" : ""}${shell.escape(Number(metric.deviation_percent).toFixed(2))}%`}</td><td>${shell.escape(metric.coverage_percent ?? "未记录")}%</td><td>${dailyTrendSvg(metric, true)}</td></tr>`).join("");
    const findingRows = findings.map((item) => `<article class="daily-finding"><span>${shell.escape(item.finding_id || "F")}</span><div><h3>${shell.escape(item.title || "巡检发现")}</h3><p>${shell.escape(reportDisplayText(item.summary || ""))}</p></div><strong>${shell.badge(item.status || item.severity || "INFO")}</strong></article>`).join("");
    const findingNoteRows = findingNotes.map((item) => `<li>${shell.escape(reportDisplayText(item))}</li>`).join("");
    const evidenceSources = (boundary.sources || []).map((item) => `<li>${shell.escape(item)}</li>`).join("");
    return `<article class="ops-report-presentation daily-formal-report"><header class="daily-report-head"><div class="daily-report-kicker">KBot AIOps · 正式运维报告</div><div class="daily-report-title-row"><div><h1>${shell.escape(header.title || data.title || report.title || "数据库日常巡检报告")}</h1><p>报告以冻结的结构化巡检证据生成；偏离只表示相对七日基线的变化。</p>${sourceLink}</div><div class="ops-actions">${actionButtons}</div></div><dl class="daily-report-metadata">${metadata}</dl></header><section class="daily-conclusion"><div><p class="daily-section-kicker">REPORT CONCLUSION</p><h2>报告结论</h2></div><p class="daily-conclusion-copy">${shell.escape(conclusion.text || "未形成报告结论。")}</p><div class="daily-conclusion-ledger">${stat("整体健康", conclusion.health_status_display || "未判定")}${stat("数据库检查覆盖率", coverage)}${stat("监控采样覆盖率", monitoringCoverage, Number(conclusion.monitoring_coverage_percent) < 80 ? "attention" : "")}${stat("需关注 / 数据缺口", `${conclusion.attention_count || 0} / ${conclusion.gap_count || 0}`)}</div></section><section class="daily-report-section"><header><div><p class="daily-section-kicker">BASELINE DEVIATION</p><h2>关键指标与过去 7 日基线偏离</h2></div><p>偏离百分比 =（当前值 − 七日同口径均值）÷ 七日均值</p></header><div class="daily-deviation-list">${metrics.map(dailyDeviationHtml).join("") || '<p class="daily-empty">当前报告没有监控指标证据。</p>'}</div></section><section class="daily-report-section daily-trends"><header><div><p class="daily-section-kicker">SEVEN-DAY TREND</p><h2>七日趋势 · 小倍数</h2></div><p>各序列换算为七日均值指数 100，共享阅读口径。</p></header><div class="daily-trend-grid">${trendCards}</div></section>${capacityRows ? `<section class="daily-report-section"><header><div><p class="daily-section-kicker">CAPACITY DETAIL</p><h2>七日趋势与容量明细</h2></div><p>同类指标保持一行内可比。</p></header><div class="daily-table-wrap"><table class="daily-report-table"><thead><tr><th>容量指标</th><th>当前值</th><th>七日均值</th><th>偏离</th><th>覆盖率</th><th>七日趋势</th></tr></thead><tbody>${capacityRows}</tbody></table></div></section>` : ""}<section class="daily-report-section"><header><div><p class="daily-section-kicker">VERIFIED FINDINGS</p><h2>发现问题</h2></div><p>仅列经规则或证据边界确认需要关注的项目。</p></header><div class="daily-findings">${findingRows || '<p class="daily-empty">未发现需要单列的问题。</p>'}</div>${findingNoteRows ? `<ul class="daily-notes">${findingNoteRows}</ul>` : ""}</section><section class="daily-report-section"><header><div><p class="daily-section-kicker">RECOMMENDATIONS</p><h2>处置建议</h2></div><p>建议应回指发现或数据缺口，并给出复核条件。</p></header><ol class="daily-notes">${recommendations.map((item) => `<li>${shell.escape(reportDisplayText(item))}</li>`).join("") || "<li>当前没有新增处置建议。</li>"}</ol></section><section class="daily-report-section"><header><div><p class="daily-section-kicker">DATA GAPS</p><h2>数据缺口</h2></div><p>无数据不等于异常；缺口与业务健康结论分开呈现。</p></header><ol class="daily-notes daily-gaps">${gaps.map((item) => `<li>${shell.escape(reportDisplayText(item))}</li>`).join("") || "<li>未发现额外的数据缺口。</li>"}</ol></section><section class="daily-evidence"><h2>证据边界</h2><p>${shell.escape(boundary.methodology || "以冻结证据为准。")}</p>${evidenceSources ? `<ol>${evidenceSources}</ol>` : ""}<div class="daily-version-row"><label>历史版本 <select data-report-version>${versionItems}</select></label><p>人工编辑只修改文字章节，结构化指标、七日基线和证据边界保持冻结。</p></div></section><footer class="daily-report-footer"><span>KBot AIOps · 正式运维报告</span><span>${shell.escape(header.report_number || report.report_key || "")} · v${shell.escape(header.report_version || report.report_version || 1)}</span></footer></article>`;
  }

  function monthlyTrendSvg(metric) {
    const values = (metric.daily_points || []).map((item) => Number(item.value)).filter(Number.isFinite);
    if (values.length < 2) return '<span class="daily-no-baseline">有效天数不足</span>';
    const low = Math.min(...values);
    const high = Math.max(...values);
    const span = Math.max(high - low, 1e-9);
    const width = 150;
    const height = 42;
    const points = values.map((value, index) => `${(index * width / (values.length - 1)).toFixed(1)},${(height - 4 - ((value - low) / span) * (height - 8)).toFixed(1)}`).join(" ");
    return `<svg class="daily-trend-svg monthly-trend-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${shell.escape(metric.display_name || "指标")}月度趋势"><polyline points="${points}"/></svg>`;
  }

  function monthlyInspectionReportHtml(data, report, versions) {
    const monthly = data.monthly_inspection || {};
    const header = monthly.header || {};
    const conclusion = monthly.conclusion || {};
    const comparison = monthly.period_comparison || {};
    const coverage = monthly.coverage || {};
    const metrics = Array.isArray(monthly.metrics) ? monthly.metrics : [];
    const capacity = Array.isArray(monthly.capacity_rows) ? monthly.capacity_rows : [];
    const issues = Array.isArray(monthly.issues) ? monthly.issues : [];
    const recommendations = Array.isArray(monthly.recommendations) ? monthly.recommendations : [];
    const gaps = Array.isArray(monthly.gaps) ? monthly.gaps : [];
    const extensions = Array.isArray(monthly.extensions) ? monthly.extensions : [];
    const boundary = monthly.evidence_boundary || {};
    const canEdit = String(versions?.items?.[0]?.report_id || "") === String(report.report_id);
    const versionItems = (versions?.items || []).map((item) => `<option value="${shell.escape(item.report_id)}" ${String(item.report_id) === String(report.report_id) ? "selected" : ""}>v${shell.escape(item.report_version)} · ${shell.escape(shell.fmt(item.published_at))}</option>`).join("");
    const actionButtons = `${canEdit ? '<button type="button" data-write-ready data-edit-report>编辑报告</button>' : ""}${canManageOperationsKnowledge ? '<button type="button" data-write-ready data-extract-case>提炼为诊断案例</button>' : ""}<button class="primary" type="button" data-write-ready data-download-report>下载 PDF</button>`;
    const metadata = [
      ["报告时间", shell.fmt(header.report_time)], ["统计周期", `${shell.fmt(header.period_start)} 至 ${shell.fmt(header.period_end)}`],
      ["运维范围", header.scope_name || "未记录"], ["范围类型", header.scope_type === "BUSINESS_SYSTEM" ? "业务系统" : "运维目标"],
      ["目标数量", `${header.target_count || 1} 个`], ["数据库类型", (header.database_types || []).join("、") || "未记录"],
      ["报告版本", `v${header.report_version || 1}`], ["报告编号", header.report_number || "未记录"],
    ].map(([label, value]) => `<div><dt>${shell.escape(label)}</dt><dd>${shell.escape(value)}</dd></div>`).join("");
    const comparisonText = comparison.status === "AVAILABLE"
      ? `新增 ${comparison.new_issues?.length || 0} 项 · 持续 ${comparison.persistent_issues?.length || 0} 项 · 已解决 ${comparison.resolved_issues?.length || 0} 项`
      : (comparison.status_display || "无上月报告");
    const metricRows = metrics.map((metric) => `<tr><td><strong>${shell.escape(metric.display_name || "指标")}</strong><small>${shell.escape(metric.dimension_label || "全部范围")}</small></td><td>${shell.escape(metric.monthly_average_display || "无有效值")}</td><td>${shell.escape(metric.p95_display || "无有效值")}</td><td>${shell.escape(metric.minimum_display || "-")} / ${shell.escape(metric.maximum_display || "-")}</td><td>${metric.change_percent == null ? "不可计算" : `${Number(metric.change_percent) >= 0 ? "+" : ""}${shell.escape(Number(metric.change_percent).toFixed(2))}%`}</td><td>${shell.escape(metric.effective_days || 0)} / ${shell.escape(metric.expected_days || 0)}</td><td>${monthlyTrendSvg(metric)}</td></tr>`).join("");
    const capacityRows = capacity.map((metric) => `<tr><td><strong>${shell.escape(metric.display_name || "容量指标")}</strong><small>${shell.escape(metric.dimension_label || "全部范围")}</small></td><td>${shell.escape(metric.first_display || "-")}</td><td>${shell.escape(metric.last_display || "-")}</td><td>${shell.escape(metric.net_change_display || "-")}</td><td>${metric.change_percent == null ? "不可计算" : `${Number(metric.change_percent) >= 0 ? "+" : ""}${shell.escape(Number(metric.change_percent).toFixed(2))}%`}</td><td>${shell.escape(metric.coverage_percent ?? "未记录")}%</td></tr>`).join("");
    const issueRows = issues.map((item) => `<article class="daily-finding"><span>${shell.escape(item.finding_id || "F")}</span><div><h3>${shell.escape(item.title || "巡检发现")}</h3><p>${shell.escape(reportDisplayText(item.summary || ""))}</p><small>${shell.escape(item.first_seen || "未记录")} 至 ${shell.escape(item.last_seen || "未记录")} · 影响 ${shell.escape(item.affected_days || 0)} 天 · 出现 ${shell.escape(item.occurrence_count || 0)} 次</small></div><strong>${shell.badge(item.severity || "INFO")}</strong></article>`).join("");
    const extensionRows = extensions.map((item) => `<li><strong>${shell.escape(item.extension_key || "扩展")}</strong> · ${shell.escape(item.status === "RESERVED" ? "接口已预留，等待后续模块接入" : item.status || "未配置")}</li>`).join("");
    return `<article class="ops-report-presentation daily-formal-report monthly-formal-report"><header class="daily-report-head"><div class="daily-report-kicker">KBot AIOps · 正式月度运维报告</div><div class="daily-report-title-row"><div><h1>${shell.escape(header.title || "数据库月度巡检报告")}</h1><p>按完整自然月汇总；生成日瞬时值不进入月度结论。</p></div><div class="ops-actions">${actionButtons}</div></div><dl class="daily-report-metadata">${metadata}</dl></header><section class="daily-conclusion"><div><p class="daily-section-kicker">MONTHLY CONCLUSION</p><h2>月度结论</h2></div><p class="daily-conclusion-copy">${shell.escape(conclusion.text || "未形成月度结论。")}</p><div class="daily-conclusion-ledger"><div><span>完成巡检</span><strong>${shell.escape(coverage.completed_run_count || 0)} 次</strong></div><div><span>指标日覆盖率</span><strong>${shell.escape(coverage.metric_day_coverage_percent ?? "未记录")}%</strong></div><div><span>问题总数</span><strong>${shell.escape(conclusion.issue_count || 0)} 项</strong></div><div><span>上月对比</span><strong>${shell.escape(comparison.status_display || "无上月报告")}</strong></div></div></section><section class="daily-report-section"><header><div><p class="daily-section-kicker">MONTH-OVER-MONTH</p><h2>上月对比</h2></div><p>仅比较相同指标、维度和单位；对比结果随本报告冻结。</p></header><p class="monthly-comparison-copy">${shell.escape(comparisonText)}</p></section><section class="daily-report-section"><header><div><p class="daily-section-kicker">MONTHLY METRICS</p><h2>月度关键指标</h2></div><p>月均、P95、峰谷、月内变化与有效天数共同呈现。</p></header><div class="daily-table-wrap"><table class="daily-report-table monthly-report-table"><thead><tr><th>指标</th><th>月均值</th><th>P95</th><th>最小 / 最大</th><th>月内变化</th><th>有效天数</th><th>月度趋势</th></tr></thead><tbody>${metricRows || '<tr><td colspan="7">当前自然月没有可统计的指标日点。</td></tr>'}</tbody></table></div></section>${capacityRows ? `<section class="daily-report-section"><header><div><p class="daily-section-kicker">CAPACITY & GROWTH</p><h2>容量与增长</h2></div><p>容量结论以月初、月末和月内变化为主，不使用生成日当前值。</p></header><div class="daily-table-wrap"><table class="daily-report-table"><thead><tr><th>容量指标</th><th>月初</th><th>月末</th><th>净变化</th><th>变化率</th><th>覆盖率</th></tr></thead><tbody>${capacityRows}</tbody></table></div></section>` : ""}<section class="daily-report-section"><header><div><p class="daily-section-kicker">DATABASE CHECKS</p><h2>数据库专项健康</h2></div><p>汇总本月结构化数据库检查，不把无数据直接判定为异常。</p></header><p>${shell.escape(Object.entries(monthly.database_checks?.status_counts || {}).map(([key, value]) => `${reportDisplayText(key)} ${value} 项`).join(" · ") || "本月没有可汇总的数据库专项检查。")}</p></section><section class="daily-report-section"><header><div><p class="daily-section-kicker">ISSUE LEDGER</p><h2>问题台账</h2></div><p>按稳定指纹归并出现次数、影响天数和生命周期。</p></header><div class="daily-findings">${issueRows || '<p class="daily-empty">本月未形成需跟踪的问题。</p>'}</div></section><section class="daily-report-section"><header><div><p class="daily-section-kicker">NEXT ACTIONS</p><h2>处置建议与下月重点</h2></div></header><ol class="daily-notes">${recommendations.map((item) => `<li>${shell.escape(reportDisplayText(item))}</li>`).join("") || "<li>当前没有新增处置建议。</li>"}</ol></section><section class="daily-report-section"><header><div><p class="daily-section-kicker">DATA GAPS</p><h2>数据缺口</h2></div></header><ol class="daily-notes daily-gaps">${gaps.map((item) => `<li>${shell.escape(reportDisplayText(item))}</li>`).join("") || "<li>未发现额外的数据缺口。</li>"}</ol></section><section class="daily-report-section"><header><div><p class="daily-section-kicker">EXTENSIONS</p><h2>扩展接口状态</h2></div></header><ul class="daily-notes">${extensionRows || "<li>没有扩展模块。</li>"}</ul></section><section class="daily-evidence"><h2>证据边界</h2><p>${shell.escape(boundary.methodology || "以冻结证据为准。")}</p><div class="daily-version-row"><label>历史版本 <select data-report-version>${versionItems}</select></label><p>结构化月度统计和上月对比随报告冻结。</p></div></section><footer class="daily-report-footer"><span>KBot AIOps · 正式月度运维报告</span><span>${shell.escape(header.report_number || "")} · v${shell.escape(header.report_version || 1)}</span></footer></article>`;
  }

  function reportPresentationHtml(data, report, versions, target) {
    if (data?.monthly_inspection?.schema_version === "MONTHLY_INSPECTION_PRESENTATION.v1") {
      return monthlyInspectionReportHtml(data, report, versions, target);
    }
    if (data?.daily_inspection?.schema_version === "DAILY_INSPECTION_PRESENTATION.v1") {
      return dailyInspectionReportHtml(data, report, versions, target);
    }
    const sections = Array.isArray(data.sections) ? data.sections : [];
    const canEdit = String(versions?.items?.[0]?.report_id || "") === String(report.report_id);
    const versionItems = (versions?.items || []).map((item) => `<option value="${shell.escape(item.report_id)}" ${String(item.report_id) === String(report.report_id) ? "selected" : ""}>v${shell.escape(item.report_version)} · ${shell.escape(shell.fmt(item.published_at))}</option>`).join("");
    const sourceLink = report.source_conversation_id
      ? `<a href="./chat.html?conversation=${encodeURIComponent(report.source_conversation_id)}">查看来源会话</a>` : "";
    return `<article class="ops-report-presentation"><header class="ops-head"><div><h2 data-report-title>${shell.escape(data.title || report.title || "正式报告")}</h2><p>${shell.escape(data.template?.display_name || "报告模板")} · ${shell.escape(report.template_ref || data.template?.template_ref || "系统模板")} · 模板 v${shell.escape(report.template_version || data.template?.version || "1")} · 报告 v${shell.escape(report.report_version)} · ${shell.escape(data.status_display || reportDisplayText(data.status || "UNKNOWN"))}</p>${sourceLink}</div><div class="ops-actions">${canEdit ? '<button type="button" data-write-ready data-edit-report>编辑报告</button>' : ""}${canManageOperationsKnowledge ? '<button type="button" data-write-ready data-extract-case>提炼为诊断案例</button>' : ""}<button class="primary" type="button" data-write-ready data-download-report>下载 PDF</button></div></header>${leadershipBriefingHtml(data.leadership_briefing)}${reportMetadataHtml(data, report, target)}<section class="ops-panel"><div class="ops-panel-body"><label>历史版本 <select data-report-version>${versionItems}</select></label><p>历史版本可随时预览和重新下载；人工编辑会创建新版本，不会覆盖旧版。</p></div></section>${sections.map((section) => `<section class="ops-panel" data-report-section="${shell.escape(section.kind || "")}"><div class="ops-panel-head"><h3>${shell.escape(section.display_name || reportDisplayText(section.kind || "章节"))}${section.human_edited ? " · 人工编辑" : ""}</h3></div><div class="ops-panel-body" data-report-section-body><ul>${(section.items || []).map((item) => `<li>${shell.escape(reportDisplayText(item))}</li>`).join("")}</ul></div></section>`).join("")}</article>`;
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
    const report = await KBotAIOpsAuth.request(appApi + `/reports/${encodedId}`);
    const [data, versions, target] = await Promise.all([
      KBotAIOpsAuth.request(appApi + `/reports/${encodedId}/presentation`),
      KBotAIOpsAuth.request(appApi + `/reports/${encodedId}/versions`),
      KBotAIOpsAuth.request(appApi + `/targets/${encodeURIComponent(report.target_id)}`).catch(() => null),
    ]);
    panel.innerHTML = reportPresentationHtml(data, report, versions, target);
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
    return `<div class="ops-panel-head source-overview-head"><div><p>${shell.escape(sourceTypeLabels[source.source_type] || source.source_type || "监控源")}</p><h2>${shell.escape(source.display_name)}</h2><small>${shell.escape(source.adapter_id)} Adapter · v${shell.escape(source.adapter_version)}</small></div><div class="ops-actions">${shell.badge(source.status)}${shell.badge(source.connectivity_check_pending ? "CHECKING" : source.connectivity_status)}</div></div><div class="ops-panel-body source-overview-grid"><section><h3>接入信息</h3><dl class="ops-detail">${row("监控源类型", sourceTypeLabels[source.source_type] || source.source_type)}${row("访问地址", endpoint)}${row("Adapter", `${source.adapter_id} · v${source.adapter_version}`)}${row("Loki 租户", tenant)}</dl></section><section><h3>连接与健康</h3><dl class="ops-detail">${row("启用状态", statusLabels[source.status] || source.status)}${row("连接状态", connectivity)}${row("最近检查", shell.fmt(source.last_connectivity_check_at))}${row("最近成功", shell.fmt(source.last_connectivity_success_at))}${row("最近错误", source.last_error_code || "无")}</dl></section><section><h3>凭据与 Webhook</h3><dl class="ops-detail">${row("访问凭据", credential(source.secret))}${row("Webhook 验签凭据", credential(source.webhook_secret))}${row("Webhook Key", source.webhook_configured ? "已生成" : "未生成")}${row("TLS Profile", credential(source.tls_profile))}</dl></section><section><h3>管理信息</h3><dl class="ops-detail">${row("监控源 ID", source.source_id)}${row("创建时间", shell.fmt(source.created_at))}${row("创建人", source.created_by)}${row("最后更新", shell.fmt(source.updated_at))}${row("更新人", source.updated_by)}</dl></section><section class="source-capability-section"><h3>系统声明能力</h3><p>由当前监控源类型和内置 Adapter 决定。</p><div class="ops-capability-list">${capabilities(source.declared_capabilities, "当前 Adapter 未声明能力")}</div></section><section class="source-capability-section"><h3>最近验证能力</h3><p>来自最近一次成功的连通性检查，不代表尚未验证的能力不可用。</p><div class="ops-capability-list">${capabilities(source.discovered_capabilities, "尚未通过连通性检查验证能力")}</div></section></div><div class="source-detail-note">具体监控 Label 与数据库对象的映射统一在“运维目标编辑”中维护。</div>`;
  }

  async function initializeTargetMonitorMappings(targetId, target) {
    if (!document.getElementById("target-monitor-binding-form")) {
      const summaryList = document.getElementById("target-monitor-binding-list");
      const summaryState = document.getElementById("target-monitor-mapping-state");
      if (!summaryList || !summaryState) return;
      try {
        const [sourcePage, rows] = await Promise.all([
          KBotAIOpsAuth.request(`${appApi}/diagnostic-sources?limit=200`),
          KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/source-bindings`),
        ]);
        const sources = Array.isArray(sourcePage) ? sourcePage : sourcePage.items || [];
        const sourceNames = new Map(sources.map((source) => [String(source.source_id), source]));
        const bindings = Array.isArray(rows) ? rows : [];
        const activeCount = bindings.filter((binding) => binding.status === "ACTIVE").length;
        summaryState.textContent = activeCount ? `${activeCount} 条有效映射` : "未配置";
        summaryState.className = `ops-badge ${activeCount ? "good" : "bad"}`;
        summaryList.innerHTML = bindings.length
          ? `<table class="ops-table"><thead><tr><th>监控源</th><th>类型</th><th>数据库 Label</th><th>主机 Label</th><th>状态</th></tr></thead><tbody>${bindings.map((binding) => {
            const source = sourceNames.get(String(binding.source_id));
            return `<tr><td>${shell.escape(source?.display_name || shell.short(binding.source_id))}</td><td>${shell.escape(source?.source_type || "—")}</td><td><code>${shell.escape(binding.locator_hint || "未配置")}</code></td><td><code>${shell.escape(binding.host_locator_hint || "未配置")}</code></td><td>${shell.badge(binding.status)}</td></tr>`;
          }).join("")}</tbody></table>`
          : '<div class="ops-error">尚未绑定监控 Label；Target 当前不能启用。</div>';
      } catch (error) {
        summaryList.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`;
        summaryState.textContent = "读取失败";
        summaryState.className = "ops-badge bad";
      }
      return;
    }
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
        ? `<table class="ops-table"><thead><tr><th>监控源</th><th>类型</th><th>Label / 外部标识</th><th>状态</th><th>操作</th></tr></thead><tbody>${bindings.map((binding) => {
          const source = sourceById(binding.source_id);
          const hostKey = binding.host_locator_hint;
          return `<tr><td><strong>${shell.escape(source?.display_name || shell.short(binding.source_id))}</strong></td><td>${shell.escape(source?.source_type || "—")}</td><td>数据库 <code>${shell.escape(binding.source_locator_key || binding.locator_hint)}</code><br>主机 <code>${shell.escape(hostKey || "未配置")}</code></td><td>${shell.badge(binding.status)}</td><td><button type="button" class="danger" data-delete-source-binding="${shell.escape(binding.binding_id)}" data-row-version="${binding.row_version}">删除</button></td></tr>`;
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
        ? `<div class="ops-empty">已绑定数据库 Label：<code>${shell.escape(binding.source_locator_key || binding.locator_hint)}</code>；本次仅修改主机 Label。</div>`
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
            ? `<div class="target-monitor-candidates">${candidates.map((candidate, index) => `<label class="agent-switch-row"><input type="radio" name="monitor_candidate_ref" value="${shell.escape(candidate.candidate_ref)}" ${index === 0 ? "checked" : ""}><span><strong>${shell.escape(candidate.display_name)}</strong><small>${shell.escape(candidate.db_type)} · <code>${shell.escape(candidate.locator_key)}</code></small></span></label>`).join("")}</div>`
            : '<div class="ops-empty">没有尚未映射且与当前数据库类型一致的候选 Label。</div>';
        } else {
          candidateList.innerHTML = `<div class="ops-empty">已绑定数据库 Label：<code>${shell.escape(binding.source_locator_key || binding.locator_hint)}</code>；本次仅修改主机 Label。</div>`;
        }
        const hostCandidates = page.host_items || [];
        hostCandidateList.innerHTML = hostCandidates.length
          ? `<div class="target-monitor-candidates">${hostCandidates.map((candidate, index) => `<label class="agent-switch-row"><input type="radio" name="monitor_host_candidate_ref" value="${shell.escape(candidate.candidate_ref)}" ${index === 0 ? "checked" : ""}><span><strong>${shell.escape(candidate.display_name)}</strong><small>主机 · <code>${shell.escape(candidate.locator_key)}</code></small></span></label>`).join("")}</div>`
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
    bindingList.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-delete-source-binding]");
      if (!button || !confirm("确认删除这条监控映射吗？历史告警会保留；删除最后一条有效映射时 Target 将自动停用。")) return;
      const binding = bindings.find((item) => item.binding_id === button.dataset.deleteSourceBinding);
      if (!binding) return;
      const leavesNoActiveBinding = !bindings.some((item) => item.status === "ACTIVE" && item.binding_id !== binding.binding_id);
      button.disabled = true;
      result.textContent = "正在删除监控映射…";
      result.dataset.tone = "";
      try {
        await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/source-bindings/${encodeURIComponent(binding.binding_id)}`, {
          method: "DELETE",
          headers: {
            "If-Match": `"rv-${binding.row_version}"`,
            "Idempotency-Key": KBotAIOpsAuth.uuid(),
          },
        });
        await reloadBindings();
        result.textContent = leavesNoActiveBinding && target.status === "ENABLED"
          ? "监控映射已删除；Target 因无有效监控映射已自动停用。"
          : "监控映射已删除。";
        result.dataset.tone = "good";
      } catch (error) {
        result.textContent = error.message;
        result.dataset.tone = "bad";
        button.disabled = false;
      }
    });
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
          const businessSystemId = new URLSearchParams(location.search).get("business_system_id");
          editTargetAccess.href = `targets.html?edit=${encodeURIComponent(id)}${businessSystemId ? `&business_system_id=${encodeURIComponent(businessSystemId)}` : ""}`;
        }
        await initializeTargetSubscription(id, data);
        await initializeTargetMonitorMappings(id, data);
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
    initializeTargetFacts,
    reload() {
      const page = document.body.dataset.page;
      return configs[page] ? renderList(page) : Promise.resolve();
    },
  };
})();
