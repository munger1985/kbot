/* 代表参数的专用保护入口；不显示原值，不通过普通附件或浏览器存储保存材料。 */
(function () {
  "use strict";
  function sourceLabel(source) {
    return `${source.database_type} · ${JSON.stringify(source.subject)} · ${source.step_id} 第 ${source.row_index + 1} 行`;
  }
  function mount(panel, run, appApi) {
    if (!["COMPLETED", "PARTIAL"].includes(run.status)) return;
    const escape = KBotAIOpsShell.escape;
    const root = document.createElement("details");
    root.className = "ops-panel-body";
    root.innerHTML = `<summary>代表参数保护材料</summary>
      <p>仅通过可信 HTTPS 或受保护隧道提交真实参数，不要经公网明文 HTTP 传输。</p>
      <p>此入口只加密保存参数和声明的解析环境，不触发 EXPLAIN 或业务 SQL 执行。独立风险审批及计划采集尚未接通；材料不证明参数代表性或原环境。不要通过聊天、普通附件提交参数原值。</p>
      <form class="ops-form" autocomplete="off">
        <div class="ops-field span-12"><label>已取证的 SQL<select name="source"><option value="">请选择完整统计主体</option></select></label></div>
        <div class="ops-field span-6"><label>专用参数 JSON 文件<input name="material" type="file" accept=".json,application/json"></label></div>
        <div class="ops-field span-6"><label>有效期<select name="ttl"><option value="300">5分钟</option><option value="900" selected>15分钟</option><option value="1800">30分钟</option></select></label></div>
        <div class="ops-filter-actions"><button type="submit">保护保存，不执行</button><button type="button" data-material-refresh>刷新状态</button></div>
      </form>
      <details><summary>专用文件格式</summary><p>值为 null、布尔、字符串或有界数组；数字、日期以精确字符串填写。database_type_name 必须显式声明。环境未知填写 null，不猜测生产环境。</p><pre data-material-example></pre></details>
      <div data-material-list></div><p>到期即不可再使用；刷新状态会清除已到期的在线密文。平台备份按原留存策略处理。</p>`;
    panel.appendChild(root);
    root.querySelector("[data-material-example]").textContent = JSON.stringify({
      schema_version: "SQL_PLAN_PARAMETER_MATERIAL.v1", parameters: [{ position: 1, database_type_name: "pg_catalog.int8", value: "123" }],
      execution_context: { database_name: null, schema_name: null, role_name: null },
    }, null, 2);
    const form = root.querySelector("form");
    const selector = form.elements.source;
    const input = form.elements.material;
    const list = root.querySelector("[data-material-list]");
    const base = appApi + "/runs/" + encodeURIComponent(run.ops_run_id);
    let sources = [];
    let busy = false;
    function setBusy(value) {
      busy = value;
      form.querySelectorAll("button, input, select").forEach((element) => { element.disabled = value; });
      list.querySelectorAll("button").forEach((element) => { element.disabled = value; });
    }
    const active = () => document.contains(root);
    async function load() {
      setBusy(true);
      try {
        const [sourceResult, materialResult] = await Promise.allSettled([
          KBotAIOpsAuth.request(base + "/sql-plan-material-sources", { cache: "no-store" }),
          KBotAIOpsAuth.request(base + "/sql-plan-materials", { cache: "no-store" }),
        ]);
        if (!active()) return;
        sources = sourceResult.status === "fulfilled" ? sourceResult.value : [];
        selector.innerHTML = '<option value="">请选择完整统计主体</option>' + sources.map((source, index) => `<option value="${index}">${escape(sourceLabel(source))}</option>`).join("");
        if (materialResult.status === "rejected") throw materialResult.reason;
        const labels = { STORED: "保护保存，未授权执行", EXPIRED: "已过期", REVOKED: "已撤回", UNAVAILABLE: "来源或授权已变化" };
        list.innerHTML = (materialResult.value.items || []).map((material) => `<p>${escape(material.parameter_count)}个参数 · ${escape(labels[material.state] || material.state)} · 到期 ${escape(material.expires_at)}
          ${material.state === "REVOKED" ? "" : `<button type="button" data-revoke-material="${escape(material.material_id)}">撤回并清除在线密文</button>`}</p>`).join("") || "<p>尚未保存材料。</p>";
        if (!sources.length) list.insertAdjacentHTML("afterbegin", "<p>没有可用的完整 SQL 统计行；目标或权限变化后仍可撤回已有材料。</p>");
      } catch (_) { if (active()) KBotAIOpsShell.toast("无法读取计划材料状态，请刷新后重试"); }
      finally { if (active()) setBusy(false); }
    }
    root.addEventListener("toggle", () => { if (root.open && !busy) void load(); else if (!root.open) input.value = ""; });
    root.querySelector("[data-material-refresh]").onclick = load;
    form.onsubmit = async (event) => {
      event.preventDefault();
      if (busy) return;
      const source = sources[Number(selector.value)];
      const file = input.files?.[0];
      if (selector.value === "" || !source || !file || file.size > 65536) {
        KBotAIOpsShell.toast("请选择已取证的 SQL 和不超过64 KiB的专用 JSON 参数文件"); return;
      }
      const ttlSeconds = Number(form.elements.ttl.value);
      setBusy(true);
      try {
        const protectedDocument = await file.text();
        if (!active()) return;
        await KBotAIOpsAuth.request(base + "/sql-plan-materials", { method: "POST",
          body: JSON.stringify({ submission_id: KBotAIOpsAuth.uuid7(), source_artifact_id: source.source_artifact_id,
            step_id: source.step_id, row_index: source.row_index, protected_document: protectedDocument, ttl_seconds: ttlSeconds }) });
        if (!active()) return;
        KBotAIOpsShell.toast("参数材料已保护保存，未触发计划采集或业务 SQL 执行");
        await load();
      } catch (_) { if (active()) KBotAIOpsShell.toast("参数材料未保存，请刷新状态后重试"); }
      finally { input.value = ""; if (active()) setBusy(false); }
    };
    list.onclick = async (event) => {
      const button = event.target.closest("[data-revoke-material]");
      if (!button || busy) return;
      setBusy(true);
      try {
        await KBotAIOpsAuth.request(base + "/sql-plan-materials/" + encodeURIComponent(button.dataset.revokeMaterial), { method: "DELETE" });
        if (!active()) return;
        KBotAIOpsShell.toast("参数材料已撤回，在线密文已清除");
        await load();
      } catch (_) { if (active()) KBotAIOpsShell.toast("无法撤回参数材料"); }
      finally { if (active()) setBusy(false); }
    };
  }
  window.KBotAIOpsPlanMaterials = { mount };
})();
