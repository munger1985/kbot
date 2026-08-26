(function () {
  "use strict";
  const appApi = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let sourceReloadTimer = null;
  let sourceReloadAttempts = 0;
  const configs = {
    situations: { path: "/situations", cols: [["title", "情境"], ["severity", "严重度", "badge"], ["status", "状态", "badge"], ["event_count", "事件数"], ["last_observed_at", "最近观测", "date"]] },
    runs: { path: "/runs", cols: [["ops_run_id", "运行 ID", "id"], ["trigger_type", "触发方式"], ["investigation_mode", "模式"], ["status", "状态", "badge"], ["created_at", "创建时间", "date"]], detail: "run-detail.html?id=" },
    reports: { path: "/reports", cols: [["report_key", "报告"], ["report_type", "类型"], ["status", "状态", "badge"], ["summary", "摘要"], ["period_end", "周期结束", "date"]], detail: "report-detail.html?id=" },
    inspections: { path: "/inspection-fires", cols: [["fire_id", "执行 ID", "id"], ["scheduled_at", "计划时间", "date"], ["status", "状态", "badge"], ["target_count", "目标数"], ["failed_count", "失败数"]] },
    changes: { path: "/proposals", cols: [["proposal_id", "建议 ID", "id"], ["action_template_id", "动作模板"], ["risk", "风险", "badge"], ["status", "状态", "badge"], ["expires_at", "失效时间", "date"]] },
    targets: { path: "/targets", cols: [["display_name", "目标"], ["db_type", "数据库"], ["status", "启用状态", "badge"], ["connectivity_status", "连通性", "badge"], ["observed_status", "观测状态", "badge"], ["updated_at", "更新时间", "date"], ["_actions", "操作", "target-actions"]], detail: "target-detail.html?id=" },
    "diagnostic-sources": { path: "/diagnostic-sources", cols: [["display_name", "诊断源"], ["source_type", "类型"], ["status", "启用状态", "badge"], ["connectivity_status", "连通性", "badge"], ["updated_at", "更新时间", "date"], ["_actions", "操作", "source-actions"]], detail: "diagnostic-source-detail.html?id=" },
    policies: { path: "/policies", cols: [["display_name", "策略"], ["policy_key", "策略标识"], ["status", "状态", "badge"], ["version_no", "规则版本"]], detail: "#" },
    "inspection-plans": { path: "/inspection-plans", cols: [["display_name", "计划"], ["schedule_type", "调度类型"], ["timezone", "时区"], ["status", "状态", "badge"], ["updated_at", "更新时间", "date"]], detail: "inspection-plan-detail.html?id=" },
    "notification-subscriptions": { path: "/notification-subscriptions", cols: [["target_id", "目标", "id"], ["channel_type", "渠道"], ["minimum_severity", "最低级别", "badge"], ["enabled", "启用"]] },
  };
  const resourceId = (item) => item.ops_run_id || item.report_id || item.target_id || item.source_id || item.policy_id || item.plan_id;
  function cell(item, [key, , type]) {
    const value = item?.[key];
    if (type === "source-actions") {
      const checking = item.connectivity_check_pending;
      const healthButton = `<button type="button" data-source-action="connectivity" ${checking ? "disabled" : ""}>${checking ? "检查中" : "检查连通性"}</button>`;
      const lifecycleButton = item.status === "ENABLED"
        ? '<button type="button" data-source-action="disable">停用</button>'
        : ["CONNECTED", "DEGRADED"].includes(item.connectivity_status) && !checking
          ? '<button type="button" class="primary" data-source-action="enable">启用</button>'
          : "";
      return `<div class="ops-actions">${healthButton}${lifecycleButton}</div>`;
    }
    if (type === "target-actions") {
      const checking = item.connectivity_check_pending;
      const checkButton = `<button type="button" data-target-action="connectivity" ${checking ? "disabled" : ""}>${checking ? "检查中" : "检查连通性"}</button>`;
      const buttons = item.status === "ENABLED"
        ? ['<button type="button" data-target-action="disable">停用</button>']
        : ["CONNECTED", "DEGRADED"].includes(item.connectivity_status)
          ? ['<button type="button" class="primary" data-target-action="enable">启用</button>']
          : [];
      return `<div class="ops-actions">${checkButton}${buttons.join("")}</div>`;
    }
    if (key === "connectivity_status" && item.connectivity_check_pending) {
      return shell.badge("检查中");
    }
    if (type === "badge") return shell.badge(value);
    if (type === "date") return shell.escape(shell.fmt(value));
    if (type === "id") return `<code>${shell.escape(shell.short(value))}</code>`;
    return shell.escape(value ?? "—");
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
    const path = action === "connectivity"
      ? `/diagnostic-sources/${sourceId}/connectivity-checks`
      : `/diagnostic-sources/${sourceId}/${action}`;
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(appApi + path, {
        method: "POST",
        headers: {
          "If-Match": `"rv-${item.row_version}"`,
          "Idempotency-Key": KBotAIOpsAuth.uuid(),
        },
        body: JSON.stringify({}),
      });
      shell.toast(action === "connectivity" ? "连通性检查已提交" : action === "enable" ? "诊断源已启用" : "诊断源已停用");
      await renderList("diagnostic-sources");
      if (action === "connectivity") {
        sourceReloadAttempts = 0;
        scheduleSourceReload();
      }
    } catch (error) {
      shell.toast(error.message);
      button.disabled = false;
    }
  }
  async function runTargetAction(button, item) {
    const action = button.dataset.targetAction;
    const targetId = encodeURIComponent(item.target_id);
    const path = action === "connectivity"
      ? `/targets/${targetId}/connectivity-checks`
      : `/targets/${targetId}/${action}`;
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(
        appApi + path,
        {
          method: "POST",
          headers: {
            "If-Match": `"rv-${item.row_version}"`,
            "Idempotency-Key": KBotAIOpsAuth.uuid(),
          },
          body: JSON.stringify({}),
        },
      );
      const messages = {
        connectivity: "Target 连通性检查已提交",
        enable: "运维目标已启用",
        disable: "运维目标已停用",
      };
      shell.toast(messages[action]);
      await renderList("targets");
    } catch (error) {
      shell.toast(error.message);
      button.disabled = false;
    }
  }
  async function renderList(page) {
    const cfg = configs[page];
    const head = document.getElementById("ops-table-head");
    const body = document.getElementById("ops-table-body");
    head.innerHTML = `<tr>${cfg.cols.map((col) => `<th>${col[1]}</th>`).join("")}</tr>`;
    try {
      const payload = await KBotAIOpsAuth.request(appApi + cfg.path);
      const items = Array.isArray(payload) ? payload : payload?.items || [];
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
            "diagnostic-sources": globalThis.KBotAIOpsConfigurations,
            policies: globalThis.KBotAIOpsConfigurations,
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
    } catch (error) {
      body.innerHTML = `<tr><td class="ops-empty" colspan="${cfg.cols.length}">${shell.escape(error.message)}</td></tr>`;
    }
  }
  async function renderDetail(page) {
    const id = new URLSearchParams(location.search).get("id");
    const paths = { "run-detail": "/runs/", "report-detail": "/reports/", "target-detail": "/targets/", "diagnostic-source-detail": "/diagnostic-sources/", "inspection-plan-detail": "/inspection-plans/" };
    const panel = document.getElementById("ops-detail");
    if (!id) { panel.innerHTML = '<div class="ops-error">URL 缺少资源 id</div>'; return; }
    try {
      const data = await KBotAIOpsAuth.request(appApi + paths[page] + encodeURIComponent(id));
      panel.innerHTML = `<dl class="ops-detail">${Object.entries(data).filter(([, value]) => typeof value !== "object").map(([key, value]) => `<dt>${shell.escape(key)}</dt><dd>${shell.escape(value ?? "—")}</dd>`).join("")}</dl><pre class="ops-code">${shell.escape(JSON.stringify(data, null, 2))}</pre>`;
    } catch (error) { panel.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`; }
  }
  async function renderDashboard() {
    const paths = ["/situations?status=OPEN&limit=5", "/runs?limit=5", "/proposals?status=PENDING_APPROVAL&limit=5", "/reports?limit=5"];
    const results = await Promise.allSettled(paths.map((path) => KBotAIOpsAuth.request(appApi + path)));
    results.forEach((result, index) => { document.querySelector(`[data-metric="${index}"]`).textContent = result.status === "fulfilled" ? result.value?.items?.length ?? 0 : "—"; });
    const rows = results.flatMap((result, index) => result.status === "fulfilled" ? (result.value?.items || []).map((value) => ({ type: ["故障情境", "诊断运行", "变更待办", "报告"][index], value })) : []);
    document.getElementById("dashboard-stream").innerHTML = rows.length ? rows.slice(0, 12).map(({ type, value }) => `<tr><td>${type}</td><td>${shell.escape(value.title || value.summary || value.status || "—")}</td><td>${shell.fmt(value.last_observed_at || value.created_at || value.period_end)}</td></tr>`).join("") : '<tr><td class="ops-empty" colspan="3">当前范围内暂无活动</td></tr>';
  }
  async function renderSimple(page) {
    const paths = { agents: `${appApi}/agents`, "report-templates": `${appApi}/report-templates`, "api-clients": `${appApi}/api-clients`, notifications: "/api/v1/notifications" };
    const panel = document.getElementById("ops-simple");
    if (!paths[page]) return;
    try { panel.innerHTML = `<pre class="ops-code">${shell.escape(JSON.stringify(await KBotAIOpsAuth.request(paths[page]), null, 2))}</pre>`; }
    catch (error) { panel.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`; }
  }
  shell.ready.then(() => {
    document.querySelectorAll("header.ops-head button:not([onclick]):not([data-write-ready])").forEach((button) => {
      button.disabled = true;
      button.title = "该写操作将在对应配置表单接入后开放";
    });
    const page = document.body.dataset.page;
    if (configs[page]) renderList(page);
    else if (page.endsWith("-detail")) renderDetail(page);
    else if (page === "dashboard") renderDashboard();
    else renderSimple(page);
  });
  globalThis.KBotAIOpsPages = {
    reload() {
      const page = document.body.dataset.page;
      return configs[page] ? renderList(page) : Promise.resolve();
    },
  };
})();
