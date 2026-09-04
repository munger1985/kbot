(function () {
  "use strict";
  const appApi = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let sourceReloadTimer = null;
  let sourceReloadAttempts = 0;
  const configs = {
    targets: { path: "/targets", cols: [["display_name", "目标"], ["db_type", "数据库"], ["status", "启用状态", "badge"], ["connectivity_status", "连通性", "badge"], ["observed_status", "观测状态", "badge"], ["updated_at", "更新时间", "date"], ["_actions", "操作", "target-actions"]], detail: "target-detail.html?id=" },
    "diagnostic-sources": { path: "/diagnostic-sources", cols: [["display_name", "诊断源"], ["source_type", "类型"], ["status", "启用状态", "badge"], ["connectivity_status", "连通性", "badge"], ["updated_at", "更新时间", "date"], ["_actions", "操作", "source-actions"]], detail: "diagnostic-source-detail.html?id=" },
    "inspection-plans": { path: "/inspection-plans", cols: [["display_name", "计划"], ["agent_name", "DBA Agent"], ["schedule_type", "调度周期", "schedule"], ["timezone", "时区"], ["status", "状态", "badge"], ["updated_at", "更新时间", "date"], ["_actions", "操作", "inspection-actions"]], detail: "inspection-plan-detail.html?id=" },
    reports: { path: "/reports", cols: [["title", "报告"], ["report_type", "类型"], ["report_version", "当前版本"], ["status", "状态", "badge"], ["period_end", "报告截止时间", "date"], ["summary", "摘要"]], detail: "report-detail.html?id=" },
  };
  const resourceId = (item) => item.ops_run_id || item.report_id || item.target_id || item.source_id || item.plan_id;
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
      const detailButton = '<button type="button" data-target-action="detail">详情</button>';
      const checkButton = item.readonly_connection_enabled
        ? `<button type="button" data-target-action="connectivity" ${checking ? "disabled" : ""}>${checking ? "检查中" : "检查连通性"}</button>`
        : "";
      const buttons = item.status === "ENABLED"
        ? ['<button type="button" data-target-action="disable">停用</button>']
        : (!item.readonly_connection_enabled || ["CONNECTED", "DEGRADED"].includes(item.connectivity_status))
          ? ['<button type="button" class="primary" data-target-action="enable">启用</button>']
          : [];
      return `<div class="ops-actions">${detailButton}${checkButton}${buttons.join("")}</div>`;
    }
    if (type === "inspection-actions") {
      const action = item.status === "ACTIVE" ? "pause" : item.status === "PAUSED" ? "activate" : "";
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
    if (action === "detail") {
      location.href = `target-detail.html?id=${targetId}`;
      return;
    }
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
            "diagnostic-sources": globalThis.KBotAIOpsConfigurations,
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

  async function initializeTargetSubscription(targetId, target) {
    const elements = subscriptionElements();
    if (!elements) return;
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

  function reportPresentationHtml(data, report, versions) {
    const sections = Array.isArray(data.sections) ? data.sections : [];
    const canEdit = String(versions?.items?.[0]?.report_id || "") === String(report.report_id);
    const versionItems = (versions?.items || []).map((item) => `<option value="${shell.escape(item.report_id)}" ${String(item.report_id) === String(report.report_id) ? "selected" : ""}>v${shell.escape(item.report_version)} · ${shell.escape(shell.fmt(item.published_at))}</option>`).join("");
    return `<article class="ops-report-presentation"><header class="ops-head"><div><h2 data-report-title>${shell.escape(data.title || report.title || "正式报告")}</h2><p>${shell.escape(data.template?.display_name || "报告模板")} · ${shell.escape(data.status || "UNKNOWN")} · v${shell.escape(report.report_version)}</p></div><div class="ops-actions">${canEdit ? '<button type="button" data-write-ready data-edit-report>编辑报告</button>' : ""}<button class="primary" type="button" data-write-ready data-download-report>下载 PDF</button></div></header><section class="ops-panel"><div class="ops-panel-body"><label>历史版本 <select data-report-version>${versionItems}</select></label><p>历史版本可随时预览和重新下载；人工编辑会创建新版本，不会覆盖旧版。</p></div></section>${sections.map((section) => `<section class="ops-panel" data-report-section="${shell.escape(section.kind || "")}"><div class="ops-panel-head"><h3>${shell.escape(section.kind || "章节")}${section.human_edited ? " · 人工编辑" : ""}</h3></div><div class="ops-panel-body" data-report-section-body><ul>${(section.items || []).map((item) => `<li>${shell.escape(item)}</li>`).join("")}</ul></div></section>`).join("")}</article>`;
  }

  function beginReportEdit(panel, data, report, versions) {
    const sections = Array.isArray(data.sections) ? data.sections : [];
    const editableSections = sections.filter((section) => !protectedReportSections.has(section.kind));
    panel.innerHTML = `<article class="ops-report-presentation"><header class="ops-head"><div><label>报告标题 <input data-report-title-input maxlength="512" value="${shell.escape(data.title || report.title || "")}"></label><p>人工编辑仅修改展示文字，已冻结的证据边界和证据索引不会变化。</p></div><div class="ops-actions"><button type="button" data-write-ready data-cancel-report-edit>取消</button><button class="primary" type="button" data-write-ready data-save-report-edit>保存为新版本</button></div></header>${editableSections.map((section) => `<section class="ops-panel"><div class="ops-panel-head"><h3>${shell.escape(section.kind || "章节")}</h3></div><div class="ops-panel-body"><textarea data-report-edit-section="${shell.escape(section.kind)}" rows="6">${shell.escape((section.items || []).join("\n"))}</textarea></div></section>`).join("")}</article>`;
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
    panel.querySelector("[data-report-version]").onchange = (event) => {
      const selected = event.currentTarget.value;
      if (selected && selected !== String(report.report_id)) {
        location.href = `report-detail.html?id=${encodeURIComponent(selected)}`;
      }
    };
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
      panel.innerHTML = `<dl class="ops-detail">${Object.entries(data).filter(([, value]) => typeof value !== "object").map(([key, value]) => `<dt>${shell.escape(key)}</dt><dd>${shell.escape(value ?? "—")}</dd>`).join("")}</dl><pre class="ops-code">${shell.escape(JSON.stringify(data, null, 2))}</pre>`;
      if (page === "target-detail") await initializeTargetSubscription(id, data);
    } catch (error) { panel.innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`; }
  }
  async function renderSimple(page) {
    const paths = { "api-clients": `${appApi}/api-clients` };
    const panel = document.getElementById("ops-simple");
    if (!paths[page] || !panel) return;
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
    else if (page !== "agents") renderSimple(page);
  });
  globalThis.KBotAIOpsPages = {
    reload() {
      const page = document.body.dataset.page;
      return configs[page] ? renderList(page) : Promise.resolve();
    },
  };
})();
