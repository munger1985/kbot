/* 执行前冻结业务条件；数值与效果等级只由服务端从受控证据复算。 */
(() => {
  const states = new WeakMap();
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const roles = { PRIMARY: "主要效果", GUARDRAIL: "护栏", LOAD: "负载可比性", SYMPTOM: "症状恢复" };
  const comparisons = { DECREASE_BY: "至少下降指定比例", INCREASE_BY: "至少上升指定比例", STABLE_WITHIN: "相对变化不超过", AT_MOST: "不高于绝对值", AT_LEAST: "不低于绝对值" };
  const options = (items, selected) => Object.entries(items).map(([key, label]) => `<option value="${key}"${key === selected ? " selected" : ""}>${label}</option>`).join("");
  const reportOptions = (reports, placeholder = "请选择已生成的工作负载报告") => `<option value="">${placeholder}</option>${reports.map(item => `<option value="${esc(item.artifact_id)}">${esc(item.period_start || "")} — ${esc(item.period_end || "")} · ${esc(item.status || "")}</option>`).join("")}`;
  const metricLabel = item => `${item.label || item.metric}${item.statement_identity?.statement_identity_value ? ` · SQL ${item.statement_identity.statement_identity_value} · 库 ${item.statement_identity.database_name} · 用户 ${item.statement_identity.user_identifier} · 顶层 ${item.statement_identity.top_level ?? "未知"}${item.statement_identity.plan_hash_value ? ` · 计划 ${item.statement_identity.plan_hash_value}` : ""}` : ""}`;
  function lockControls(root) {
    root.dataset.effectBusy = "true";
    const controls = [...root.querySelectorAll("button")].map(button=>({button,disabled:button.disabled}));
    controls.forEach(({button})=>{button.disabled=true;});
    return () => {
      delete root.dataset.effectBusy;
      controls.forEach(({button,disabled})=>{if(button.isConnected)button.disabled=disabled;});
    };
  }

  function updateMetrics(root) {
    const state = states.get(root);
    const id = root.querySelector('[name="baseline_artifact_id"]').value;
    const metrics = state.reports.find(item => item.artifact_id === id)?.metrics || [];
    root.querySelectorAll('[data-rule="metric"]').forEach(select => {
      const selected = select.value;
      select.innerHTML = `<option value="">请选择实际证据中的指标</option>${metrics.map(item => `<option value="${esc(JSON.stringify({metric: item.metric, statement_identity: item.statement_identity}))}">${esc(metricLabel(item))}</option>`).join("")}`;
      select.value = [...select.options].some(option => option.value === selected) ? selected : "";
    });
    const controlId = root.querySelector('[name="control_baseline_artifact_id"]')?.value;
    const controlMetrics = state.controlReports.find(item => item.artifact_id === controlId)?.metrics || [];
    root.querySelectorAll('[data-rule="control"]').forEach(select => {
      const selected = select.value;
      select.innerHTML = `<option value="">选择同口径指标及完整主体</option>${controlMetrics.map(item => `<option value="${esc(JSON.stringify({metric: item.metric, statement_identity: item.statement_identity}))}">${esc(metricLabel(item))}</option>`).join("")}`;
      select.value = [...select.options].some(option => option.value === selected) ? selected : "";
    });
    const notice = root.querySelector("[data-effect-truncated]");
    if (notice) notice.hidden = !state.reports.find(item => item.artifact_id === id)?.metrics_truncated;
  }
  function addRule(root, role = "GUARDRAIL", initial = false) {
    const rules = root.querySelector("[data-effect-rules]");
    if (rules.children.length >= 32) return;
    const row = document.createElement("div");
    row.className = "effect-rule";
    row.dataset.effectRule = "";
    row.innerHTML = `<label>条件名称<input data-rule="key" value="${initial ? {PRIMARY:"primary",GUARDRAIL:"guardrail",LOAD:"load"}[role] : `signal_${rules.children.length + 1}`}" required maxlength="128"></label><label>用途<select data-rule="role">${options(roles, role)}</select></label><label>证据指标<select data-rule="metric" required></select></label><label>成功条件<select data-rule="comparison">${options(comparisons, role === "PRIMARY" ? "DECREASE_BY" : role === "LOAD" ? "STABLE_WITHIN" : "AT_MOST")}</select></label><label>阈值（比例为 0–1）<input data-rule="threshold" value="${role === "PRIMARY" ? "0.2" : role === "LOAD" ? "0.1" : ""}" required maxlength="50" inputmode="decimal"></label><button type="button" class="text-button" data-effect-remove>移除此条件</button>`;
    row.querySelector("[data-effect-remove]").onclick = () => row.remove();
    rules.append(row);
    const label = document.createElement("label");
    label.dataset.controlRule = "";
    label.innerHTML = '对照组对应指标<select data-rule="control"></select>';
    row.querySelector('[data-rule="metric"]').closest("label").after(label);
    const enabled = root.querySelector('[name="controlled_comparison"]')?.checked;
    label.hidden = !enabled;
    label.querySelector("select").disabled = !enabled;
    label.querySelector("select").required = Boolean(enabled);
    updateMetrics(root);
  }
  async function load(root, api, request, toast) {
    const id = root.dataset.effectProposal;
    if (root.dataset.effectBusy) return;
    root.dataset.effectBusy = "true";
    const oldForm = root.querySelector("[data-effect-freeze],[data-effect-evaluate]");
    const draft = oldForm ? [...oldForm.querySelectorAll("[name]")].map(input=>({name:input.name,value:input.value,checked:input.checked})) : [];
    const oldRules = oldForm ? [...oldForm.querySelectorAll("[data-effect-rule]")].map(row=>[...row.querySelectorAll("[data-rule]")].map(input=>({key:input.dataset.rule,value:input.value}))) : null;
    const controlled = root.querySelector('[name="controlled_comparison"]')?.checked ?? root.dataset.effectControlled === "true";
    root.dataset.effectControlled = String(controlled);
    const existingButtons = [...root.querySelectorAll("button")].map(button=>({button,disabled:button.disabled}));
    existingButtons.forEach(({button})=>{button.disabled=true;});
    try {
    const selectedControl = root.dataset.effectControlTarget;
    const query = selectedControl ? `?control_target_id=${encodeURIComponent(selectedControl)}` : "";
    const [view, proposal] = await Promise.all([request(`${api}/proposals/${encodeURIComponent(id)}/effect${query}`), request(`${api}/proposals/${encodeURIComponent(id)}`)]);
    const controlTarget = view.frozen_plan?.business_effect?.controlled_comparison?.target_id || selectedControl;
    if (controlTarget) {root.dataset.effectControlTarget = controlTarget;root.dataset.effectControlled = "true";}
    const hasControl = controlled || Boolean(controlTarget);
    const reports = view.evidence_candidates.filter(item => item.schema_version === "AIOPS_WORKLOAD_REPORT.v2" && item.target_id === proposal.target_id);
    const controlReports = view.evidence_candidates.filter(item => item.schema_version === "AIOPS_WORKLOAD_REPORT.v2" && item.target_id === controlTarget);
    const verifications = view.evidence_candidates.filter(item => item.schema_version === "ACTION_VERIFICATION.v1");
    states.set(root, { reports, controlReports, controlTarget });
    const frozen = Boolean(view.frozen_plan?.business_effect);
    const refresh = '<button class="secondary-button" type="button" data-effect-refresh>刷新受控证据</button>';
    const body = root.querySelector("[data-effect-body]");
    if (!frozen) {
      body.innerHTML = `${refresh}<form data-effect-freeze><label>基线窗口<select name="baseline_artifact_id" required>${reportOptions(reports)}</select></label>${!reports.length ? "<p>尚无可用窗口报告。请先采集并生成当前 Target 的工作负载报告；累计统计不能代替窗口基线。</p>" : ""}<div data-effect-rules></div><button type="button" class="secondary-button" data-effect-add>添加验证条件</button><label>处置后稳定等待（秒）<input name="settle_delay_seconds" type="number" min="0" max="86400" value="60" required></label><label>处置后观察窗口（秒）<input name="after_window_seconds" type="number" min="60" max="604800" value="900" required></label><label>停止条件（每行一项）<textarea name="stop_conditions" rows="3" maxlength="16000" required>护栏退化或出现新业务错误时停止，按已审批的回退方案处理</textarea></label><button class="primary-button" type="submit">冻结执行前验证计划</button></form>`;
      root.querySelector('[name="baseline_artifact_id"]').value = reports[0]?.artifact_id || "";
      for (const role of ["PRIMARY", "GUARDRAIL", "LOAD"]) addRule(root, role, true);
      root.querySelector('[name="baseline_artifact_id"]').onchange = () => updateMetrics(root);
      root.querySelector("[data-effect-add]").onclick = () => addRule(root);
      const automatic = document.createElement("label");
      automatic.innerHTML = '<span><input type="checkbox" name="auto_evaluate" checked> 动作复核后自动验证业务效果</span><small>仅使用已启用的周期采集，窗口结束后自动生成报告；不会修改目标库或开启采集。</small>';
      root.querySelector('[name="stop_conditions"]').closest("label").before(automatic);
      const control = document.createElement("div");
      control.style.display = "contents";
      control.innerHTML = `<label><span><input type="checkbox" name="controlled_comparison"${controlTarget ? " checked" : ""}> 预登记匹配对照验证</span><small>需要另一个已授权、物理身份不同的同类数据库，以及至少三对等长、同步且不重叠的前期窗口。不会执行实验变更；对照支持不等于根因已确认。</small></label><fieldset data-control-plan${controlTarget ? "" : " hidden"}><legend>执行前对照组</legend><label>对照 Target ID<input name="control_target_id" value="${esc(controlTarget || "")}" maxlength="36"></label><button type="button" class="secondary-button" data-control-load>读取对照目标的受控证据</button><label>同步对照基线<select name="control_baseline_artifact_id">${reportOptions(controlReports,"请选择与处理组基线同步的窗口")}</select></label>${[1, 2].map(index => `<div><label>前期窗口 ${index} · 处理组<select name="prior_treatment_${index}">${reportOptions(reports,"按时间从早到晚选择")}</select></label><label>前期窗口 ${index} · 对照组<select name="prior_control_${index}">${reportOptions(controlReports,"选择同步对照窗口")}</select></label></div>`).join("")}<p>服务端会检查前期趋势、最少执行样本、窗口覆盖及逐信号口径；不能事后挑选对照或改变成功阈值。</p></fieldset>`;
      root.querySelector('[data-effect-rules]').before(control);
      const toggleControl = () => {
        const enabled = root.querySelector('[name="controlled_comparison"]').checked;
        root.dataset.effectControlled = String(enabled);
        root.querySelector('[data-control-plan]').hidden = !enabled;
        root.querySelectorAll('[data-control-plan] input,[data-control-plan] select,[data-rule="control"]').forEach(input => { input.disabled = !enabled; input.required = enabled; });
        root.querySelectorAll('[data-control-rule]').forEach(label => { label.hidden = !enabled; });
        const windowForm = root.querySelector('[data-effect-window]');
        if (windowForm) {
          const target = windowForm.querySelector('[name="window_target"]');
          if (!enabled) target?.closest("label").remove();
          else if (!target) windowForm.insertAdjacentHTML("afterbegin",'<label>采集目标<select name="window_target"><option value="treatment">处理组</option><option value="control">对照组</option></select></label>');
        }
      };
      root.querySelector('[name="controlled_comparison"]').checked = hasControl;
      toggleControl();
      root.querySelector('[name="controlled_comparison"]').onchange = toggleControl;
      root.querySelector('[name="control_baseline_artifact_id"]').onchange = () => updateMetrics(root);
      root.querySelector('[data-control-load]').onclick = () => {
        root.dataset.effectControlTarget = root.querySelector('[name="control_target_id"]').value.trim();
        load(root, api, request, toast).catch(error => toast(error.message));
      };
    } else {
      const values = Object.entries(view.result?.measurements || {});
      const result = values.length ? `<div class="table-scroll"><table><thead><tr><th>条件</th><th>基线</th><th>处置后</th><th>单位</th><th>是否满足</th></tr></thead><tbody>${values.map(([key, item]) => `<tr><td>${esc(key)}</td><td>${esc(item.baseline ?? "未知")}</td><td>${esc(item.after ?? "未知")}</td><td>${esc(item.unit ?? "未知")}</td><td>${item.passed === true ? "满足" : item.passed === false ? "不满足" : "证据不足"}</td></tr>`).join("")}</tbody></table></div>` : "";
      body.innerHTML = `${refresh}<p>验证条件已冻结，不能根据执行结果事后挑选指标。可在详情中核对原条件；重复验证沿用同一计划。</p><details data-frozen-details><summary>冻结条件详情</summary><pre>${esc(JSON.stringify(view.frozen_plan, null, 2))}</pre></details><form data-effect-evaluate><label>处置后窗口<select name="after_artifact_id" required>${reportOptions(reports,"请选择处置后的受控工作负载报告")}</select></label><label>动作只读复核<select name="verification_artifact_id" required><option value="">请选择当前 Proposal 的复核结果</option>${verifications.map(item => `<option value="${esc(item.artifact_id)}">${esc(item.status)} · ${esc(item.artifact_id)}</option>`).join("")}</select></label><button type="submit" class="primary-button">复算业务效果并生成报告</button></form>${view.result?.measured_effect ? `<p>实测效果：${esc(view.result.measured_effect)} · 护栏：${esc(view.result.guardrails_passed)} · 归因：${esc(view.result.attribution_strength)}</p>` : ""}<span data-effect-result style="display:contents">${result}${view.result?.gap_codes?.length ? `<p>缺口：${esc(view.result.gap_codes.join("、"))}</p>` : ""}</span>`;
    }
    if (frozen && hasControl) {
      const label = document.createElement("label");
      label.innerHTML = `同步对照后窗口<select name="control_after_artifact_id" required>${reportOptions(controlReports,"请选择与处理组后窗口同步的对照报告")}</select>`;
      root.querySelector('[name="after_artifact_id"]').closest("label").after(label);
    }
    if (frozen && view.result?.control_analysis) {
      const detail = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "匹配对照复算及边界";
      const text = document.createElement("pre");
      text.textContent = JSON.stringify(view.result.control_analysis, null, 2);
      const boundary = document.createElement("p");
      boundary.textContent = "对照支持本次措施效果，不等于随机试验或已确认根因，不能完全排除未观测并行变更。";
      detail.append(summary, text, boundary);
      body.querySelector("[data-effect-result]").before(detail);
    }
    root.querySelector("[data-effect-refresh]").onclick = () => load(root, api, request, toast).catch(error => toast(error.message));
    if (frozen && view.followup?.status) {
      const status = document.createElement("p");
      const label = { SCHEDULED: "等待观察窗口结束", COMPLETED: "已完成", FAILED: "未完成" };
      status.textContent = `自动验证：${label[view.followup.status] || "未知"}${view.followup.available_at ? ` · 计划处理时间 ${view.followup.available_at}` : ""}${view.followup.error_code ? ` · ${view.followup.error_code}` : ""}`;
      body.querySelector("[data-frozen-details]").before(status);
    }
    const notice = document.createElement("p");
    notice.dataset.effectTruncated = "";
    notice.hidden = true;
    notice.textContent = "当前只展示该报告前 128 个指标选项，未展示的指标不视为不存在。";
    if (!frozen) root.querySelector("[data-effect-rules]").before(notice);
    if (frozen && view.report_id) {
      const link = document.createElement("a");
      link.href = `./report-detail.html?id=${encodeURIComponent(view.report_id)}`;
      link.textContent = "查看业务效果验证报告";
      link.className = "row-action";
      body.querySelector("[data-effect-result]").before(link);
    }
    const generator = document.createElement("details");
    const localTime = offset => { const date = new Date(Date.now() + offset); return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16); };
    const datePicker = (name,value) => `<span class="date-time-picker-control"><span class="date-time-picker-display" aria-hidden="true">${value.replace("T"," ")}:00</span><input class="date-time-picker-native" name="${name}" type="datetime-local" value="${value}" required><span class="date-time-picker-icon" aria-hidden="true"></span></span>`;
    generator.innerHTML = `<summary>从周期快照生成窗口证据</summary><form data-effect-window><label>窗口开始（本地时间）${datePicker("window_start",localTime(-15 * 60000))}</label><label>窗口结束（本地时间）${datePicker("window_end",localTime(0))}</label><button class="secondary-button" type="submit">生成此窗口报告</button><p>Oracle 使用平台周期快照，不冒充 AWR；快照不足时不能补造历史。</p></form>`;
    body.querySelector("[data-effect-refresh]").after(generator);
    generator.querySelectorAll(".date-time-picker-native").forEach(input => {input.oninput=()=>{input.previousElementSibling.textContent=input.value ? input.value.replace("T"," ")+(input.value.length===16?":00":"") : "请选择日期和时间";};});
    if (hasControl) {
      const label = document.createElement("label");
      label.innerHTML = '采集目标<select name="window_target"><option value="treatment">处理组</option><option value="control">对照组</option></select>';
      generator.querySelector("form").prepend(label);
    }
    generator.querySelector("form").onsubmit = async event => {
      event.preventDefault();
      if (root.dataset.effectBusy) return;
      const unlock = lockControls(root);
      let result;
      const data = new FormData(event.target);
      const isControl = data.get("window_target") === "control";
      try {
        result = await request(`${api}/reports/workload`, { method: "POST", headers: { "Idempotency-Key": crypto.randomUUID() }, body: JSON.stringify({ target_id: isControl ? root.querySelector('[name="control_target_id"]')?.value || controlTarget : proposal.target_id, period_start: new Date(data.get("window_start")).toISOString(), period_end: new Date(data.get("window_end")).toISOString() }) });
      } catch (error) { toast(error.message); }
      finally { unlock(); }
      if (result) {
        await load(root, api, request, toast);
        const select = root.querySelector(isControl ? (frozen ? '[name="control_after_artifact_id"]' : '[name="control_baseline_artifact_id"]') : (frozen ? '[name="after_artifact_id"]' : '[name="baseline_artifact_id"]'));
        select.value = result.artifact_id;
        select.dispatchEvent(new Event("change", {bubbles:true}));
        if (!frozen) updateMetrics(root);
        toast("已从周期快照生成窗口证据，未执行目标库变更");
      }
    };
    const form = root.querySelector("[data-effect-freeze],[data-effect-evaluate]");
    for (const item of draft) {
      const input = form.elements[item.name];
      if (input) {input.value=item.value;if(input.type==="checkbox")input.checked=item.checked;}
    }
    if (!frozen && oldRules) {
      const container = root.querySelector("[data-effect-rules]");
      container.replaceChildren();
      for (const _rule of oldRules) addRule(root);
      [...container.children].forEach((row,index)=>{
        for(const item of oldRules[index]) {
          const input=row.querySelector(`[data-rule="${item.key}"]`);
          if(input) input.value=item.value;
        }
      });
      updateMetrics(root);
    }
    const updateSubmit = () => {
      form.querySelector('[type="submit"]').disabled = Boolean(root.dataset.effectBusy) || (frozen
        ? !form.elements.after_artifact_id.value || !form.elements.verification_artifact_id.value
        : !form.elements.baseline_artifact_id.value);
      const control = form.querySelector("[data-control-load]");
      if (control) control.disabled = Boolean(root.dataset.effectBusy) || !form.elements.control_target_id.value.trim() || form.elements.control_target_id.value.trim() === proposal.target_id;
    };
    form.addEventListener("input", updateSubmit);
    form.addEventListener("change", updateSubmit);
    updateSubmit();
    form.onsubmit = async event => {
      event.preventDefault();
      if (root.dataset.effectBusy) return;
      const unlock = lockControls(root);
      let succeeded = false;
      try {
        const data = new FormData(form);
        const payload = frozen ? { expected_row_version: proposal.row_version, after_artifact_id: data.get("after_artifact_id"), verification_artifact_id: data.get("verification_artifact_id") } : {
          expected_row_version: proposal.row_version, baseline_artifact_id: data.get("baseline_artifact_id"), settle_delay_seconds: Number(data.get("settle_delay_seconds")), after_window_seconds: Number(data.get("after_window_seconds")), stop_conditions: String(data.get("stop_conditions")).split("\n").map(item => item.trim()).filter(Boolean),
          auto_evaluate: data.has("auto_evaluate"),
          controlled_comparison: null,
          signals: [...form.querySelectorAll("[data-effect-rule]")].map(row => ({ key: row.querySelector('[data-rule="key"]').value, role: row.querySelector('[data-rule="role"]').value, comparison: row.querySelector('[data-rule="comparison"]').value, threshold: row.querySelector('[data-rule="threshold"]').value, ...JSON.parse(row.querySelector('[data-rule="metric"]').value) })),
        };
        if (!frozen && data.has("controlled_comparison")) {
          payload.controlled_comparison = {
            target_id: data.get("control_target_id"), baseline_artifact_id: data.get("control_baseline_artifact_id"),
            prior_pairs: [1, 2].map(index => ({ treatment_artifact_id: data.get(`prior_treatment_${index}`), control_artifact_id: data.get(`prior_control_${index}`) })),
            signals: payload.signals.map((rule, index) => ({ ...rule, ...JSON.parse(form.querySelectorAll('[data-rule="control"]')[index].value) })),
          };
        }
        if (frozen && controlTarget) payload.control_after_artifact_id = data.get("control_after_artifact_id");
        await request(`${api}/proposals/${encodeURIComponent(id)}/${frozen ? "effect-evaluation" : "effect-plan"}`, { method: "POST", body: JSON.stringify(payload) });
        toast(frozen ? "已根据受控证据复算并生成效果报告" : "验证条件已冻结，尚未执行任何变更");
        succeeded = true;
      } catch (error) { toast(error.message); }
      finally { unlock();updateSubmit(); }
      if (succeeded) await load(root, api, request, toast);
    };
    root.dataset.effectLoaded = "true";
    } finally {
      delete root.dataset.effectBusy;
      existingButtons.forEach(({button,disabled})=>{if(button.isConnected)button.disabled=disabled;});
      root.querySelector("[data-effect-freeze],[data-effect-evaluate]")?.dispatchEvent(new Event("change"));
    }
  }
  window.AIOpsEffectVerification = { load };
})();
