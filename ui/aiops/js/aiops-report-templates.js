(function () {
  "use strict";
  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const body = () => document.getElementById("ops-report-template-body");
  const values = (form, name) => Array.from(form.querySelectorAll(`[name="${name}"]:checked`)).map((item) => item.value);

  function names(items) {
    const labels = { CHAT: "智能诊断", ALERT: "告警诊断", INSPECTION: "日常巡检", AD_HOC: "单次诊断", DAILY: "日常", MONTHLY: "月度", QUARTERLY: "季度", ANNUAL: "年度" };
    return items.map((item) => labels[item] || item).join("、") || "—";
  }

  async function load() {
    try {
      const rows = await KBotAIOpsAuth.request(`${api}/report-templates`);
      body().innerHTML = rows.map((item) => `<tr><td>${shell.escape(item.display_name)}</td><td>${shell.escape(names(item.applicable_source_kinds || []))}</td><td>${shell.escape(names(item.allowed_period_kinds || []))}</td><td>${shell.escape(item.version || item.version_no || "—")}</td><td><code>${shell.escape(shell.short(item.content_hash))}</code></td><td>${item.system_defined ? shell.badge("系统预设") : shell.badge(item.status || "ACTIVE")}</td></tr>`).join("") || '<tr><td class="ops-empty" colspan="6">暂无报告模板</td></tr>';
    } catch (error) { body().innerHTML = `<tr><td class="ops-empty" colspan="6">${shell.escape(error.message)}</td></tr>`; }
  }

  function bind() {
    const dialog = document.getElementById("report-template-dialog");
    const form = document.getElementById("report-template-form");
    document.getElementById("create-report-template").onclick = () => { form.reset(); dialog.showModal(); };
    dialog.querySelector("[data-close-template]").onclick = () => dialog.close();
    form.onsubmit = async (event) => {
      event.preventDefault();
      const sourceKinds = values(form, "source_kind");
      const periodKinds = values(form, "period_kind");
      const sections = values(form, "section");
      const result = document.getElementById("report-template-result");
      if (!sourceKinds.length || !periodKinds.length) { result.textContent = "请至少选择一个入口和一个报告周期。"; return; }
      try {
        await KBotAIOpsAuth.request(`${api}/report-templates`, { method: "POST", headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() }, body: JSON.stringify({ display_name: form.elements.display_name.value.trim(), definition: { schema_version: "REPORT_TEMPLATE.v1", applicable_source_kinds: sourceKinds, allowed_period_kinds: periodKinds, sections: sections.map((kind) => ({ kind })) } }) });
        dialog.close(); shell.toast("自定义报告模板已创建"); await load();
      } catch (error) { result.textContent = error.message; }
    };
  }
  shell.ready.then(() => { bind(); load(); });
})();
