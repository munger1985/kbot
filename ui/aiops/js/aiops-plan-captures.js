/* 独立风险审批入口；不读取参数原值，不将审批状态冒充执行许可。 */
(function () {
  "use strict";
  function mount(panel, run, appApi, options = {}) {
    if (!["COMPLETED", "PARTIAL"].includes(run.status)) return;
    const escape = KBotAIOpsShell.escape;
    const root = document.createElement("details");
    root.className = "ops-panel-body";
    root.innerHTML = `<summary>计划采集独立审批</summary>
      <p>独立审批后，由申请人显式发起后台估算采集。不执行原业务 SQL 或 ANALYZE；估算结果不证明原解析环境、参数代表性或生产实际执行。</p>
      ${options.canReview ? '<button type="button" data-review-mode>切换审批视图</button>' : ""}
      <p data-capture-availability></p><form class="ops-form" data-capture-request>
      <label>通过静态预检的保护材料<select name="capture_material"><option value="">请选择材料</option></select></label>
      <label>单语句时限（秒）<input name="capture_timeout" type="number" min="1" max="5" value="5" required></label>
      <button type="submit">申请独立审批，不执行</button></form>
      <div data-capture-review hidden><p>当前为审批视图；默认要求另一位操作人审阅。参数原值不可见，请结合本轮原始 SQL 证据核对主体。</p>
      <label><input type="checkbox" name="ack_estimate">确认只采集估算计划，不授权 ANALYZE 或业务执行</label>
      <label data-oracle-ack><input type="checkbox" name="ack_oracle">确认 Oracle 估算需要 PLAN_TABLE 临时事务写入及回滚，且不证明参数窥探</label>
      <label data-native-bind-ack><input type="checkbox" name="ack_native_binds">允许在保护执行器内读取并比较该精确Oracle游标的绑定捕获值；不返回原值，不证明同次执行或总体代表性</label></div>
      <div data-capture-list></div><button type="button" data-capture-refresh>刷新审批状态</button>`;
    panel.appendChild(root);
    const form = root.querySelector("[data-capture-request]");
    const list = root.querySelector("[data-capture-list]");
    const selector = form.elements.capture_material;
    const ackEstimate = root.querySelector('[name="ack_estimate"]');
    const ackOracle = root.querySelector('[name="ack_oracle"]');
    const ackNativeBinds = root.querySelector('[name="ack_native_binds"]');
    function verificationMarkup(native) {
      if (!native) return "";
      const fields = { DATABASE: "原数据库", SCHEMA: "原解析Schema", ROLE: "原解析用户", SEARCH_PATH: "原search_path", OPTIMIZER_SETTINGS: "原优化器设置", PARAMETER_TYPES: "参数类型", PARAMETER_VALUES: "原生捕获值" };
      const statuses = { MATCH: "匹配", MISMATCH: "不匹配", UNKNOWN: "证据不足", UNSUPPORTED: "原生来源不支持", NOT_REQUESTED: "未授权比较", NOT_APPLICABLE: "不适用" };
      return '<section aria-label="原生环境与绑定核验"><strong>原生核验</strong>'
        + [...native.original_context, native.parameter_types, native.parameter_values].map(check => '<p>' + escape(fields[check.field] || check.field) + "：" + escape(statuses[check.status] || check.status) + "（" + escape(check.source) + "）</p>").join("")
        + "<p>已解析计划对象 " + escape(native.resolved_objects.length) + " 个；未解析 " + escape(native.unresolved_object_nodes.length) + " 个。匹配捕获样本不证明总体参数代表性，估算不证明生产实际执行。</p></section>";
    }
    const base = appApi + "/runs/" + encodeURIComponent(run.ops_run_id);
    let captures = [];
    let executions = {};
    let busy = false;
    let reviewMode = false;
    const active = () => document.contains(root);
    function executionMarkup(item) {
      if (reviewMode || !item.capture_scheduled) return "";
      const entry = executions[item.request_id];
      if (!entry) return "<p>暂不能读取采集状态或结果，请刷新；来源或访问授权变化时不会展示旧结果。</p>";
      const terminal = ["COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "EXPIRED"].includes(entry.run_status);
      const labels = { CREATED: "已创建", QUEUED: "等待执行", RUNNING: "采集中", COMPLETED: "后台任务已完成", PARTIAL: "部分完成", FAILED: "采集失败", CANCELLED: "后台任务已取消", EXPIRED: "后台任务已超时" };
      let html = `<p>${escape(entry.phase === "QUEUED" ? "等待后台领取，尚未开始采集" : entry.cancel_requested && !terminal ? "已请求取消，等待终止确认" : labels[entry.run_status] || entry.run_status)}</p>`;
      if (entry.can_cancel) html += `<button type="button" data-capture-id="${escape(item.request_id)}" data-capture-action="CANCEL">请求取消采集</button>`;
      const result = entry.result;
      if (!result) return html;
      if (result.status === "FAILED") return html + `<p>估算未取得可用结果：${escape(result.error_code)}</p>`;
      return html + `<p>估算计划 · 采集于 ${escape(result.captured_at)}。不是实际执行计划，不含参数原值或谓词。</p>
        ${item.oracle_plan_table_write ? "<p>Oracle 参数窥探未证明。</p>" : ""}
        ${verificationMarkup(result.native_verification)}
        <div class="ops-table-wrap" tabindex="0" aria-label="估算计划节点，可横向滚动"><table class="ops-table"><thead><tr><th>节点 / 父节点</th><th>操作</th><th>对象 / 索引</th><th>估算指标</th></tr></thead><tbody>
        ${result.nodes.map((node) => `<tr><td>${escape(node.node_id)} / ${escape(node.parent_id ?? "—")}</td><td>${escape(node.operation)}</td><td>${escape([node.schema_name, node.relation_name].filter(Boolean).join("."))} ${escape(node.index_name || "")}</td><td>${escape(JSON.stringify(node.metrics))}</td></tr>`).join("")}
        </tbody></table></div>`;
    }
    function setBusy(value) {
      busy = value;
      root.querySelectorAll("button, input, select").forEach((element) => { element.disabled = value; });
      list.querySelectorAll('[data-capture-action="APPROVE"]').forEach((button) => {
        const item = captures.find((entry) => entry.request_id === button.dataset.captureId);
        button.disabled = value || !ackEstimate.checked || (item.oracle_plan_table_write && !ackOracle.checked)
          || (item.native_bind_comparison && !ackNativeBinds.checked);
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
        executions = {};
        if (!reviewMode) {
          const scheduled = page.items.filter((item) => item.capture_scheduled);
          const statuses = await Promise.allSettled(scheduled.map((item) => KBotAIOpsAuth.request(
            base + "/sql-plan-captures/" + encodeURIComponent(item.request_id) + "/execution", { cache: "no-store" })));
          if (!active()) return;
          scheduled.forEach((item, index) => { executions[item.request_id] = statuses[index].status === "fulfilled" ? statuses[index].value : null; });
        }
        form.elements.capture_timeout.max = String(page.max_statement_timeout_seconds ?? 5);
        form.elements.capture_timeout.value = String(Math.min(5, page.max_statement_timeout_seconds ?? 5));
        captures = page.items;
        form.hidden = !page.can_request;
        root.querySelector("[data-capture-review]").hidden = !page.can_review;
        root.querySelector("[data-oracle-ack]").hidden = !captures.some((item) => item.oracle_plan_table_write);
        root.querySelector("[data-native-bind-ack]").hidden = !captures.some((item) => item.native_bind_comparison);
        ackNativeBinds.checked = false;
        ackEstimate.checked = false; ackOracle.checked = false;
        root.querySelector("[data-capture-availability]").textContent = ({
          AVAILABLE: "", DISABLED: "平台审批开关未开启。", POLICY_REQUIRED: "需管理员显式配置 sql_plan_capture 及允许对象范围，普通只读权限不足以授权。",
          ACCESS_CHANGED: "原诊断的目标、来源或授权已变化。",
        })[page.availability] || "";
        const eligible = materials.status === "fulfilled" ? materials.value.items.filter((item) => item.state === "STORED" && item.capture_preflight?.status === "AWAITING_DATABASE_VALIDATION") : [];
        selector.innerHTML = '<option value="">请选择材料</option>' + eligible.map((item) => `<option value="${escape(item.material_id)}">${escape(item.material_id)}</option>`).join("");
        const labels = { PENDING: "待独立审批", APPROVED: "已审批，尚未调度采集", REJECTED: "已驳回", REVOKED: "已撤回", EXPIRED: "已过期", UNAVAILABLE: "来源或授权已失效" };
        list.innerHTML = captures.map((item) => `<section class="ops-panel-body"><strong>${escape(item.capture_scheduled ? "已调度估算采集" : labels[item.state] || item.state)}</strong>
          ${item.capture_scheduled ? `<p>审批当前状态：${escape(labels[item.state] || item.state)} · 后台运行 ${escape(item.capture_run_id)}</p>` : ""}
          <p>${escape(item.source.database_type)} · ${escape(JSON.stringify(item.source.subject))}</p>
          <p>允许对象：${escape(item.allowed_objects.join("、"))} · 语句时限 ${escape(item.statement_timeout_seconds)}秒 · 审批到期 ${escape(item.expires_at)}</p>
          <p>申请人 ${escape(item.requested_by)}${item.decided_by ? " · 审批人 " + escape(item.decided_by) : ""}</p>
          ${page.can_review && item.state === "PENDING" ? `<button type="button" data-capture-id="${escape(item.request_id)}" data-capture-action="APPROVE">批准估算风险，不执行</button><button type="button" data-capture-id="${escape(item.request_id)}" data-capture-action="REJECT">驳回</button>` : ""}
          ${!page.can_review && item.state === "APPROVED" && !item.capture_scheduled ? `<button type="button" data-capture-id="${escape(item.request_id)}" data-capture-action="START">发起估算采集</button>` : ""}
          ${executionMarkup(item)}
          ${!page.can_review && ["PENDING", "APPROVED", "EXPIRED", "UNAVAILABLE"].includes(item.state) ? `<button type="button" data-capture-id="${escape(item.request_id)}" data-capture-action="REVOKE">撤回采集请求</button>` : ""}</section>`).join("");
      } catch (_) {
        if (active()) { list.textContent = "无法读取计划采集审批，请刷新后重试"; captures = []; form.hidden = true; root.querySelector("[data-capture-review]").hidden = true; }
      } finally { if (active()) setBusy(false); }
    }
    async function mutate(suffix, method, body, message = "计划采集审批状态已更新，未执行数据库 SQL") {
      setBusy(true);
      try {
        await KBotAIOpsAuth.request(base + suffix, { method, ...(body ? { body: JSON.stringify(body) } : {}) });
        if (!active()) return;
        KBotAIOpsShell.toast(message);
        await load();
      } catch (_) { if (active()) KBotAIOpsShell.toast("计划采集操作未完成，请刷新确认状态后重试"); }
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
    ackEstimate.onchange = ackOracle.onchange = ackNativeBinds.onchange = () => setBusy(busy);
    list.onclick = (event) => {
      const button = event.target.closest("[data-capture-action]");
      if (!button || busy || button.disabled) return;
      const item = captures.find((entry) => entry.request_id === button.dataset.captureId);
      const action = button.dataset.captureAction;
      const path = "/sql-plan-captures/" + encodeURIComponent(item.request_id);
      if (action === "START") { void mutate(path + "/start", "POST", { request_sha256: item.request_sha256 }, "已调度后台估算采集；执行前仍将复核当前授权"); return; }
      if (action === "CANCEL") { void mutate(path + "/cancel", "POST", undefined, "已提交取消请求；请刷新状态确认后台及数据库终止情况"); return; }
      if (action === "REVOKE") { void mutate(path, "DELETE"); return; }
      void mutate(path + "/decision", "POST", { request_sha256: item.request_sha256, decision: action,
        acknowledge_estimate_only: action === "APPROVE" && ackEstimate.checked,
        acknowledge_oracle_plan_table_write: action === "APPROVE" && item.oracle_plan_table_write && ackOracle.checked,
        acknowledge_native_bind_comparison: action === "APPROVE" && Boolean(item.native_bind_comparison) && ackNativeBinds.checked });
    };
  }
  window.KBotAIOpsPlanCaptures = { mount };
})();
