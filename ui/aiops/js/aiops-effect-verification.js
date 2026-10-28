/* 执行前冻结业务条件；数值与效果等级只由服务端从受控证据复算。 */
(() => {
  const states = new WeakMap();
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const roles = { PRIMARY: "主要效果", GUARDRAIL: "护栏", LOAD: "负载可比性", SYMPTOM: "症状恢复" };
  const comparisons = { DECREASE_BY: "至少下降指定比例", INCREASE_BY: "至少上升指定比例", STABLE_WITHIN: "相对变化不超过", AT_MOST: "不高于绝对值", AT_LEAST: "不低于绝对值" };
  const options = (items, selected) => Object.entries(items).map(([key, label]) => `<option value="${key}"${key === selected ? " selected" : ""}>${label}</option>`).join("");
  const reportOptions = reports => `<option value="">请选择受控窗口报告</option>${reports.map(item => `<option value="${esc(item.artifact_id)}">${esc(item.period_start)} — ${esc(item.period_end)} · ${esc(item.status)}</option>`).join("")}`;

  function updateMetrics(root) {
    const state = states.get(root);
    const id = root.querySelector('[name="baseline_artifact_id"]').value;
    const metrics = state.reports.find(item => item.artifact_id === id)?.metrics || [];
    root.querySelectorAll('[data-rule="metric"]').forEach(select => {
      const selected = select.value;
      select.innerHTML = `<option value="">请选择实际证据中的指标</option>${metrics.map(item => `<option value="${esc(JSON.stringify({metric: item.metric, statement_identity: item.statement_identity}))}">${esc(item.label || item.metric)} · ${esc(item.statement_identity?.statement_identity_value ? `SQL ${item.statement_identity.statement_identity_value} · 库 ${item.statement_identity.database_name} · 用户 ${item.statement_identity.user_identifier} · 顶层 ${item.statement_identity.top_level ?? "未知"}${item.statement_identity.plan_hash_value ? ` · 计划 ${item.statement_identity.plan_hash_value}` : ""}` : "")}</option>`).join("")}`;
      select.value = [...select.options].some(option => option.value === selected) ? selected : "";
    });
    const notice = root.querySelector("[data-effect-truncated]");
    if (notice) notice.hidden = !state.reports.find(item => item.artifact_id === id)?.metrics_truncated;
  }
  function addRule(root, role = "GUARDRAIL") {
    const rules = root.querySelector("[data-effect-rules]");
    if (rules.children.length >= 32) return;
    const row = document.createElement("fieldset");
    row.dataset.effectRule = "";
    row.innerHTML = `<label>条件名称<input data-rule="key" value="signal_${rules.children.length + 1}" required maxlength="128"></label><label>用途<select data-rule="role">${options(roles, role)}</select></label><label>证据指标<select data-rule="metric" required></select></label><label>成功条件<select data-rule="comparison">${options(comparisons, role === "PRIMARY" ? "DECREASE_BY" : role === "LOAD" ? "STABLE_WITHIN" : "AT_MOST")}</select></label><label>阈值（比例为 0–1）<input data-rule="threshold" value="${role === "PRIMARY" ? "0.2" : role === "LOAD" ? "0.1" : ""}" required maxlength="50" inputmode="decimal"></label><button type="button" data-effect-remove>移除此条件</button>`;
    row.querySelector("[data-effect-remove]").onclick = () => row.remove();
    rules.append(row);
    updateMetrics(root);
  }
  async function load(root, api, request, toast) {
    const id = root.dataset.effectProposal;
    const [view, proposal] = await Promise.all([request(`${api}/proposals/${encodeURIComponent(id)}/effect`), request(`${api}/proposals/${encodeURIComponent(id)}`)]);
    const reports = view.evidence_candidates.filter(item => item.schema_version === "AIOPS_WORKLOAD_REPORT.v2");
    const verifications = view.evidence_candidates.filter(item => item.schema_version === "ACTION_VERIFICATION.v1");
    states.set(root, { reports });
    const frozen = Boolean(view.frozen_plan?.business_effect);
    const refresh = '<button type="button" data-effect-refresh>刷新受控证据</button>';
    const body = root.querySelector("[data-effect-body]");
    if (!frozen) {
      body.innerHTML = `${refresh}<p>执行前冻结主要效果、护栏、负载条件和停止条件。这不会执行变更。</p><form data-effect-freeze><label>基线窗口<select name="baseline_artifact_id" required>${reportOptions(reports)}</select></label>${!reports.length ? "<p>尚无受控窗口报告，请先采集并生成该 Target 的工作负载报告；累计值不能代替窗口基线。</p>" : ""}<div data-effect-rules></div><button type="button" data-effect-add>添加验证条件</button><label>稳定等待（秒）<input name="settle_delay_seconds" type="number" min="0" max="86400" value="60" required></label><label>观察窗口（秒）<input name="after_window_seconds" type="number" min="60" max="604800" value="900" required></label><label>停止条件（每行一项）<textarea name="stop_conditions" maxlength="16000" required>护栏退化或出现新业务错误时停止，按已审批的回退方案处理</textarea></label><button class="primary" type="submit">冻结执行前验证计划</button></form>`;
      for (const role of ["PRIMARY", "GUARDRAIL", "LOAD"]) addRule(root, role);
      root.querySelector('[name="baseline_artifact_id"]').onchange = () => updateMetrics(root);
      root.querySelector("[data-effect-add]").onclick = () => addRule(root);
      const automatic = document.createElement("label");
      automatic.innerHTML = '<span><input type="checkbox" name="auto_evaluate" checked> 动作复核后自动验证业务效果</span><small>仅使用已启用的周期采集，窗口结束后自动生成报告；不会修改目标库或开启采集。</small>';
      root.querySelector('[name="stop_conditions"]').closest("label").before(automatic);
    } else {
      const values = Object.entries(view.result?.measurements || {});
      const result = values.length ? `<p>实测效果：${esc(view.result.measured_effect)} · 护栏：${esc(view.result.guardrails_passed)} · 归因：${esc(view.result.attribution_strength)}</p><div class="table-scroll"><table><thead><tr><th>条件</th><th>基线</th><th>处置后</th><th>单位</th><th>是否满足</th></tr></thead><tbody>${values.map(([key, item]) => `<tr><td>${esc(key)}</td><td>${esc(item.baseline ?? "未知")}</td><td>${esc(item.after ?? "未知")}</td><td>${esc(item.unit ?? "未知")}</td><td>${item.passed === true ? "满足" : item.passed === false ? "不满足" : "证据不足"}</td></tr>`).join("")}</tbody></table></div><p>${esc((view.result.gap_codes || []).join("、"))}</p>` : "";
      body.innerHTML = `${refresh}<p>条件已冻结。无对照实验时不能排除并行发布、缓存预热等混杂因素。</p><details><summary>冻结条件详情</summary><pre>${esc(JSON.stringify(view.frozen_plan, null, 2))}</pre></details><form data-effect-evaluate><label>处置后窗口<select name="after_artifact_id" required>${reportOptions(reports)}</select></label><label>动作只读复核<select name="verification_artifact_id" required><option value="">请选择当前 Proposal 的复核结果</option>${verifications.map(item => `<option value="${esc(item.artifact_id)}">${esc(item.status)} · ${esc(item.artifact_id)}</option>`).join("")}</select></label><button type="submit" class="primary">复算业务效果并生成报告</button></form>${result}`;
    }
    root.querySelector("[data-effect-refresh]").onclick = () => load(root, api, request, toast).catch(error => toast(error.message));
    if (view.followup?.status) {
      const status = document.createElement("p");
      const label = { SCHEDULED: "等待观察窗口结束", COMPLETED: "已完成", FAILED: "未完成" };
      status.textContent = `自动验证：${label[view.followup.status] || "未知"}${view.followup.available_at ? ` · 计划处理时间 ${view.followup.available_at}` : ""}${view.followup.error_code ? ` · ${view.followup.error_code}` : ""}`;
      body.append(status);
    }
    const notice = document.createElement("p");
    notice.dataset.effectTruncated = "";
    notice.hidden = true;
    notice.textContent = "当前只展示该报告前 128 个指标选项，未展示的指标不视为不存在。";
    body.append(notice);
    if (view.report_id) {
      const link = document.createElement("a");
      link.href = `./report-detail.html?id=${encodeURIComponent(view.report_id)}`;
      link.textContent = "查看业务效果验证报告";
      body.append(link);
    }
    const generator = document.createElement("details");
    const localTime = offset => { const date = new Date(Date.now() + offset); return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16); };
    generator.innerHTML = `<summary>从周期快照生成窗口证据</summary><form data-effect-window><label>窗口开始（本地时间）<input name="window_start" type="datetime-local" value="${localTime(-15 * 60000)}" required></label><label>窗口结束（本地时间）<input name="window_end" type="datetime-local" value="${localTime(0)}" required></label><button type="submit">生成此窗口报告</button><p>Oracle 使用平台周期快照，不冒充 AWR；快照不足时不能补造历史。</p></form>`;
    body.append(generator);
    generator.querySelector("form").onsubmit = async event => {
      event.preventDefault();
      const button = generator.querySelector("button");
      button.disabled = true;
      try {
        const data = new FormData(event.target);
        const result = await request(`${api}/reports/workload`, { method: "POST", headers: { "Idempotency-Key": crypto.randomUUID() }, body: JSON.stringify({ target_id: proposal.target_id, period_start: new Date(data.get("window_start")).toISOString(), period_end: new Date(data.get("window_end")).toISOString() }) });
        await load(root, api, request, toast);
        const select = root.querySelector(frozen ? '[name="after_artifact_id"]' : '[name="baseline_artifact_id"]');
        select.value = result.artifact_id;
        if (!frozen) updateMetrics(root);
        toast("已从周期快照生成窗口证据，未执行目标库变更");
      } catch (error) { toast(error.message); button.disabled = false; }
    };
    const form = root.querySelector("[data-effect-freeze],[data-effect-evaluate]");
    form.onsubmit = async event => {
      event.preventDefault();
      const button = form.querySelector('[type="submit"]');
      button.disabled = true;
      try {
        const data = new FormData(form);
        const payload = frozen ? { expected_row_version: proposal.row_version, after_artifact_id: data.get("after_artifact_id"), verification_artifact_id: data.get("verification_artifact_id") } : {
          expected_row_version: proposal.row_version, baseline_artifact_id: data.get("baseline_artifact_id"), settle_delay_seconds: Number(data.get("settle_delay_seconds")), after_window_seconds: Number(data.get("after_window_seconds")), stop_conditions: String(data.get("stop_conditions")).split("\n").map(item => item.trim()).filter(Boolean),
          auto_evaluate: data.has("auto_evaluate"),
          signals: [...form.querySelectorAll("[data-effect-rule]")].map(row => ({ key: row.querySelector('[data-rule="key"]').value, role: row.querySelector('[data-rule="role"]').value, comparison: row.querySelector('[data-rule="comparison"]').value, threshold: row.querySelector('[data-rule="threshold"]').value, ...JSON.parse(row.querySelector('[data-rule="metric"]').value) })),
        };
        await request(`${api}/proposals/${encodeURIComponent(id)}/${frozen ? "effect-evaluation" : "effect-plan"}`, { method: "POST", body: JSON.stringify(payload) });
        toast(frozen ? "已根据受控证据复算并生成效果报告" : "验证条件已冻结，尚未执行任何变更");
        await load(root, api, request, toast);
      } catch (error) { toast(error.message); button.disabled = false; }
    };
  }
  window.AIOpsEffectVerification = { load };
})();
