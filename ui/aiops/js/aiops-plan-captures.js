/* 独立风险审批入口；不读取参数原值，不将审批状态冒充执行许可。 */
(function () {
  "use strict";
  function mount(panel, run, appApi, options = {}) {
    if (!["COMPLETED", "PARTIAL"].includes(run.status)) return;
    const escape = KBotAIOpsShell.escape;
    const root = document.createElement("details");
    root.className = "ops-panel-body";
    root.innerHTML = `<summary>计划采集独立审批</summary>
      <p>仅审批估算计划风险。不授权 ANALYZE、业务 SQL 或数据库变更；专用执行任务尚未接通，审批后不会立即采集。</p>
      ${options.canReview ? '<button type="button" data-review-mode>切换审批视图</button>' : ""}
      <p data-capture-availability></p><form class="ops-form" data-capture-request>
      <label>通过静态预检的保护材料<select name="capture_material"><option value="">请选择材料</option></select></label>
      <label>单语句时限（秒）<input name="capture_timeout" type="number" min="1" max="5" value="5" required></label>
      <button type="submit">申请独立审批，不执行</button></form>
      <div data-capture-review hidden><p>当前为审批视图；默认要求另一位操作人审阅。参数原值不可见，请结合本轮原始 SQL 证据核对主体。</p>
      <label><input type="checkbox" name="ack_estimate">确认只采集估算计划，不授权 ANALYZE 或业务执行</label>
      <label data-oracle-ack><input type="checkbox" name="ack_oracle">确认 Oracle 估算需要 PLAN_TABLE 临时事务写入及回滚，且不证明参数窥探</label></div>
      <div data-capture-list></div><button type="button" data-capture-refresh>刷新审批状态</button>`;
    panel.appendChild(root);
    const form = root.querySelector("[data-capture-request]");
    const list = root.querySelector("[data-capture-list]");
    const selector = form.elements.capture_material;
    const ackEstimate = root.querySelector('[name="ack_estimate"]');
    const ackOracle = root.querySelector('[name="ack_oracle"]');
    const base = appApi + "/runs/" + encodeURIComponent(run.ops_run_id);
    let captures = [];
    let busy = false;
    let reviewMode = false;
    const active = () => document.contains(root);
    function setBusy(value) {
      busy = value;
      root.querySelectorAll("button, input, select").forEach((element) => { element.disabled = value; });
      list.querySelectorAll('[data-capture-action="APPROVE"]').forEach((button) => {
        const item = captures.find((entry) => entry.request_id === button.dataset.captureId);
        button.disabled = value || !ackEstimate.checked || (item.oracle_plan_table_write && !ackOracle.checked);
      });
    }
    async function load() {
      setBusy(true);
      try {
        const [result, materials] = await Promise.allSettled([
          KBotAIOpsAuth.request(base + (reviewMode ? "/sql-plan-capture-reviews" : "/sql-plan-captures"), { cache: "no-store" }),
          reviewMode ? Promise.resolve({ items: [] }) : KBotAIOpsAuth.request(base + "/sql-plan-materials", { cache: "no-store" }),
        ]);
        if (!active()) return;
        if (result.status === "rejected") throw result.reason;
        const page = result.value;
        form.elements.capture_timeout.max = String(page.max_statement_timeout_seconds ?? 5);
        form.elements.capture_timeout.value = String(Math.min(5, page.max_statement_timeout_seconds ?? 5));
        captures = page.items;
        form.hidden = !page.can_request;
        root.querySelector("[data-capture-review]").hidden = !page.can_review;
        root.querySelector("[data-oracle-ack]").hidden = !captures.some((item) => item.oracle_plan_table_write);
        ackEstimate.checked = false; ackOracle.checked = false;
        root.querySelector("[data-capture-availability]").textContent = ({
          AVAILABLE: "", DISABLED: "平台审批开关未开启。", POLICY_REQUIRED: "需管理员显式配置 sql_plan_capture 及允许对象范围，普通只读权限不足以授权。",
          ACCESS_CHANGED: "原诊断的目标、来源或授权已变化。",
        })[page.availability] || "";
        const eligible = materials.status === "fulfilled" ? materials.value.items.filter((item) => item.state === "STORED" && item.capture_preflight?.status === "AWAITING_DATABASE_VALIDATION") : [];
        selector.innerHTML = '<option value="">请选择材料</option>' + eligible.map((item) => `<option value="${escape(item.material_id)}">${escape(item.material_id)}</option>`).join("");
        const labels = { PENDING: "待独立审批", APPROVED: "已审批，尚未调度采集", REJECTED: "已驳回", REVOKED: "已撤回", EXPIRED: "已过期", UNAVAILABLE: "来源或授权已失效" };
        list.innerHTML = captures.map((item) => `<section class="ops-panel-body"><strong>${escape(labels[item.state] || item.state)}</strong>
          <p>${escape(item.source.database_type)} · ${escape(JSON.stringify(item.source.subject))}</p>
          <p>允许对象：${escape(item.allowed_objects.join("、"))} · 语句时限 ${escape(item.statement_timeout_seconds)}秒 · 审批到期 ${escape(item.expires_at)}</p>
          <p>申请人 ${escape(item.requested_by)}${item.decided_by ? " · 审批人 " + escape(item.decided_by) : ""}</p>
          ${page.can_review && item.state === "PENDING" ? `<button type="button" data-capture-id="${escape(item.request_id)}" data-capture-action="APPROVE">批准估算风险，不执行</button><button type="button" data-capture-id="${escape(item.request_id)}" data-capture-action="REJECT">驳回</button>` : ""}
          ${!page.can_review && ["PENDING", "APPROVED", "EXPIRED", "UNAVAILABLE"].includes(item.state) ? `<button type="button" data-capture-id="${escape(item.request_id)}" data-capture-action="REVOKE">撤回采集请求</button>` : ""}</section>`).join("");
      } catch (_) {
        if (active()) { list.textContent = "无法读取计划采集审批，请刷新后重试"; captures = []; form.hidden = true; root.querySelector("[data-capture-review]").hidden = true; }
      } finally { if (active()) setBusy(false); }
    }
    async function mutate(suffix, method, body) {
      setBusy(true);
      try {
        await KBotAIOpsAuth.request(base + suffix, { method, ...(body ? { body: JSON.stringify(body) } : {}) });
        if (!active()) return;
        KBotAIOpsShell.toast("计划采集审批状态已更新，未执行数据库 SQL");
        await load();
      } catch (_) { if (active()) KBotAIOpsShell.toast("计划采集审批未更新，请刷新后重试"); }
      finally { if (active()) setBusy(false); }
    }
    root.addEventListener("toggle", () => { if (root.open && !busy) void load(); });
    root.querySelector("[data-capture-refresh]").onclick = load;
    const switchButton = root.querySelector("[data-review-mode]");
    if (switchButton) switchButton.onclick = () => {
      reviewMode = !reviewMode; switchButton.textContent = reviewMode ? "返回本人申请视图" : "切换审批视图"; void load();
    };
    form.onsubmit = (event) => {
      event.preventDefault(); if (busy || !selector.value) return;
      void mutate("/sql-plan-captures", "POST", { request_id: KBotAIOpsAuth.uuid7(), material_id: selector.value,
        statement_timeout_seconds: Number(form.elements.capture_timeout.value) });
    };
    ackEstimate.onchange = ackOracle.onchange = () => setBusy(busy);
    list.onclick = (event) => {
      const button = event.target.closest("[data-capture-action]");
      if (!button || busy || button.disabled) return;
      const item = captures.find((entry) => entry.request_id === button.dataset.captureId);
      const action = button.dataset.captureAction;
      const path = "/sql-plan-captures/" + encodeURIComponent(item.request_id);
      if (action === "REVOKE") { void mutate(path, "DELETE"); return; }
      void mutate(path + "/decision", "POST", { request_sha256: item.request_sha256, decision: action,
        acknowledge_estimate_only: action === "APPROVE" && ackEstimate.checked,
        acknowledge_oracle_plan_table_write: action === "APPROVE" && item.oracle_plan_table_write && ackOracle.checked });
    };
  }
  window.KBotAIOpsPlanCaptures = { mount };
})();
