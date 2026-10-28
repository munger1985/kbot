(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/operations-knowledge";
  const shell = globalThis.KBotAIOpsShell;
  const state = { cases: [], detail: null, canManage: false, editing: false };
  const escape = (value) => shell.escape(value == null ? "" : value);
  const statusLabel = (value) => ({ PROCESSING: "提炼中", DRAFT: "待完善", REVIEW_REQUIRED: "待审核", PUBLISHED: "已发布", RETIRED: "已退役", REJECTED: "已拒绝", FAILED: "处理失败" }[value] || value || "—");
  const kindLabel = (value) => value === "VERIFIED_RESOLUTION" ? "已验证解决方案" : "诊断参考";
  const badge = (value) => shell.badge(value, statusLabel(value));

  function renderOverview(summary) {
    const items = [["待完善", summary.draft], ["待审核", summary.review_required], ["已发布", summary.published], ["处理异常", summary.processing], ["已退役", summary.retired]];
    document.getElementById("case-overview").innerHTML = items.map(([label, value]) => `<article><span>${escape(label)}</span><strong>${escape(value || 0)}</strong></article>`).join("");
  }
  function renderRows() {
    const keyword = document.getElementById("case-search").value.trim().toLowerCase();
    const items = keyword ? state.cases.filter((item) => [item.display_name, item.root_cause_summary, item.problem_class, item.database_type, item.source_report_id].some((value) => String(value || "").toLowerCase().includes(keyword))) : state.cases;
    document.getElementById("case-count").textContent = `${items.length} 项`;
    document.getElementById("case-rows").innerHTML = items.length ? items.map((item) => `<tr><td><button class="knowledge-primary-link" type="button" data-case-id="${escape(item.asset_id)}">${escape(item.display_name)}</button><small>v${escape(item.version_no)} · L${escape(item.security_level)}</small></td><td>${escape(kindLabel(item.case_kind))}<small>${escape(item.problem_class || "未分类")}</small></td><td>${escape([item.database_type, item.database_version, item.topology].filter(Boolean).join(" · ") || "通用")}</td><td><code>${escape(item.source_report_id || "—")}</code></td><td>${badge(item.status)}</td><td>${escape(shell.fmt(item.updated_at))}</td><td><button type="button" data-case-id="${escape(item.asset_id)}">查看</button></td></tr>`).join("") : '<tr><td colspan="7" class="ops-empty">当前条件下没有历史诊断案例。</td></tr>';
  }
  async function load() {
    const query = new URLSearchParams({ limit: "200" });
    const status = document.getElementById("case-status").value;
    if (status) query.set("status", status);
    const [page, summary] = await Promise.all([
      KBotAIOpsAuth.request(`${api}/cases?${query}`),
      KBotAIOpsAuth.request(`${api}/cases/overview`),
    ]);
    state.cases = page.items || [];
    renderOverview(summary);
    renderRows();
  }
  function list(value) { return Array.isArray(value) ? value : []; }
  function section(profile, key) { return profile?.[key] && typeof profile[key] === "object" ? profile[key] : {}; }
  function renderEdit(detail) {
    const profile = detail.version.profile || {};
    const signature = section(profile, "problem_signature");
    const environment = section(profile, "environment");
    const root = section(profile, "root_cause");
    const rows = (value) => list(value).map((item) => typeof item === "string" ? item : item.summary).filter(Boolean).join("\n");
    state.editing = true;
    document.getElementById("case-detail-body").innerHTML = `<form id="case-edit-form" class="knowledge-form-section case-edit-form"><header><b>✎</b><div><strong>创建案例修订</strong><small>保存后生成新版本；来源报告、原始证据和验证引用不可修改。</small></div></header><label>案例名称<input name="display_name" maxlength="256" required value="${escape(detail.asset.display_name)}"></label><label>问题现象（每行一条）<textarea name="symptoms" rows="4" required>${escape(rows(signature.normalized_symptoms))}</textarea></label><div class="case-edit-grid"><label>组件<input name="component" value="${escape(signature.component)}"></label><label>数据库类型<input name="database_type" value="${escape(environment.database_type)}"></label><label>数据库版本<input name="database_version" value="${escape(environment.database_version)}"></label><label>平台<input name="platform" value="${escape(environment.platform)}"></label><label>拓扑<input name="topology" value="${escape(environment.topology)}"></label></div><label>调查结论与原因判断<textarea name="root_cause" rows="4" required>${escape(root.summary)}</textarea></label><label>已执行动作（每行一条）<textarea name="actions" rows="4">${escape(rows(profile.actions))}</textarea></label><label>适用条件（每行一条）<textarea name="applicability" rows="3">${escape(rows(profile.applicability))}</textarea></label><label>不适用条件（每行一条）<textarea name="contraindications" rows="3">${escape(rows(profile.contraindications))}</textarea></label></form>`;
    document.getElementById("case-detail-actions").innerHTML = '<button type="button" data-action="cancel-edit">取消</button><button class="primary" type="button" data-action="save-edit">保存为新版本</button>';
  }
  function renderDetail(detail) {
    state.editing = false;
    const profile = detail.version.profile || {};
    const signature = section(profile, "problem_signature");
    const environment = section(profile, "environment");
    const root = section(profile, "root_cause");
    const verification = section(profile, "verification");
    const actions = list(profile.actions);
    const sourceRows = (detail.sources || []).map((source) => `<article class="knowledge-source-card"><div><strong>${escape(source.source_kind)}</strong><small>${escape(source.report_id || source.run_id || source.artifact_id || "未记录")}</small></div></article>`).join("");
    const reportId = (detail.sources || []).find((item) => item.source_kind === "REPORT")?.report_id;
    const reviews = (detail.reviews || []).map((item) => `<li><div><strong>${escape(item.decision)}</strong><small>${escape(item.comment || item.reviewer_id)}</small></div><time>${escape(shell.fmt(item.created_at))}</time></li>`).join("") || "<li>暂无审核记录</li>";
    document.getElementById("case-detail-eyebrow").textContent = `${kindLabel(profile.case_kind)} · v${detail.version.version_no}`;
    document.getElementById("case-detail-title").textContent = detail.asset.display_name;
    document.getElementById("case-detail-status").innerHTML = badge(detail.version.status);
    document.getElementById("case-detail-body").innerHTML = `<section class="knowledge-form-section"><header><b>01</b><div><strong>问题现象与适用范围</strong><small>用于判断案例是否适合当前环境。</small></div></header><dl class="knowledge-summary-grid"><div><dt>问题分类</dt><dd>${escape(signature.problem_class || "未记录")}</dd></div><div><dt>组件</dt><dd>${escape(signature.component || "未记录")}</dd></div><div><dt>数据库</dt><dd>${escape([environment.database_type, environment.database_version].filter(Boolean).join(" ") || "未记录")}</dd></div><div><dt>拓扑</dt><dd>${escape(environment.topology || "未记录")}</dd></div><div><dt>症状</dt><dd>${escape(list(signature.normalized_symptoms).join("；") || "未记录")}</dd></div></dl></section><section class="knowledge-form-section"><header><b>02</b><div><strong>调查结论与原因判断</strong><small>诊断参考保留已知事实、原因强度与限制，发布仍需来源核验和人工审核。</small></div></header><p>${escape(root.summary || "未记录根因")}</p></section><section class="knowledge-form-section"><header><b>03</b><div><strong>处置与验证</strong><small>人工登记恢复不等于受控验证；不同前后引用不能自动证明措施有效。</small></div></header><ol>${actions.map((item) => `<li>${escape(item.summary)}</li>`).join("") || "<li>未记录已执行动作，仅作为诊断参考。</li>"}</ol><p>验证结果：${escape(verification.result || "NOT_VERIFIED")}</p></section><section class="knowledge-form-section"><header><b>04</b><div><strong>来源追溯</strong><small>保留正式报告、运行和证据引用。</small></div></header><div class="knowledge-source-list">${sourceRows}</div>${reportId ? `<a class="ops-button" href="./report-detail.html?id=${encodeURIComponent(reportId)}">查看来源报告</a>` : ""}</section><section class="knowledge-form-section"><header><b>05</b><div><strong>版本与审核记录</strong><small>发布、拒绝、来源更正和退役均保留审计。</small></div></header><ol class="knowledge-timeline">${reviews}</ol></section>`;
    const actionsHtml = [];
    if (state.canManage && ["DRAFT", "PUBLISHED", "REJECTED", "RETIRED"].includes(detail.version.status)) actionsHtml.push('<button type="button" data-action="edit">编辑并创建新版本</button>');
    if (state.canManage && detail.version.status === "PROCESSING") actionsHtml.push('<button type="button" data-action="reconcile">立即处理</button>');
    if (state.canManage && detail.version.status === "DRAFT") actionsHtml.push('<button class="primary" type="button" data-action="submit-review">提交审核</button>');
    if (state.canManage && detail.version.status === "REVIEW_REQUIRED") actionsHtml.push('<button class="danger" type="button" data-action="reject">拒绝</button><button class="primary" type="button" data-action="publish">审核通过并发布</button>');
    if (state.canManage && detail.version.status === "PUBLISHED") actionsHtml.push('<button class="danger" type="button" data-action="retire">退役</button>');
    if (state.canManage && detail.version.status === "FAILED") actionsHtml.push('<button class="primary" type="button" data-action="retry">重试处理</button>');
    document.getElementById("case-detail-actions").innerHTML = actionsHtml.join("");
  }
  async function openCase(assetId) {
    state.detail = await KBotAIOpsAuth.request(`${api}/cases/${encodeURIComponent(assetId)}`);
    renderDetail(state.detail);
    document.getElementById("case-drawer").showModal();
  }
  async function runAction(action) {
    if (!state.detail) return;
    if (action === "edit") { renderEdit(state.detail); return; }
    if (action === "cancel-edit") { renderDetail(state.detail); return; }
    if (action === "save-edit") {
      const form = document.getElementById("case-edit-form");
      if (!form.reportValidity()) return;
      const data = new FormData(form);
      const lines = (name) => String(data.get(name) || "").split("\n").map((item) => item.trim()).filter(Boolean);
      const profile = state.detail.version.profile || {};
      const payload = {
        expected_asset_row_version: Number(state.detail.asset.row_version),
        base_version_id: state.detail.version.asset_version_id,
        display_name: String(data.get("display_name") || "").trim(),
        profile: {
          ...profile,
          problem_signature: { ...section(profile, "problem_signature"), component: String(data.get("component") || "").trim() || null, normalized_symptoms: lines("symptoms") },
          environment: { ...section(profile, "environment"), database_type: String(data.get("database_type") || "").trim() || null, database_version: String(data.get("database_version") || "").trim() || null, platform: String(data.get("platform") || "").trim() || null, topology: String(data.get("topology") || "").trim() || null },
          root_cause: { ...section(profile, "root_cause"), summary: String(data.get("root_cause") || "").trim() },
          actions: lines("actions").map((summary) => ({ summary })),
          applicability: lines("applicability"),
          contraindications: lines("contraindications"),
        },
      };
      state.detail = await KBotAIOpsAuth.request(`${api}/cases/${encodeURIComponent(state.detail.asset.asset_id)}/versions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      shell.toast("已创建新的案例草稿版本，索引完成后可提交审核");
      renderDetail(state.detail);
      await load();
      return;
    }
    const version = state.detail.version;
    const options = action === "reconcile" ? { method: "POST" } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ expected_row_version: Number(version.row_version), comment: null }) };
    await KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(version.asset_version_id)}:${action}`, options);
    shell.toast("案例状态已更新");
    document.getElementById("case-drawer").close();
    await load();
  }
  shell.ready.then(async (access) => {
    state.canManage = new Set(access.permissions || []).has("aiops:knowledge_manage");
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    document.getElementById("case-search").oninput = renderRows;
    document.getElementById("case-status").onchange = () => load().catch((error) => shell.toast(error.message));
    document.getElementById("refresh-cases").onclick = () => load().catch((error) => shell.toast(error.message));
    document.getElementById("case-rows").onclick = (event) => { const button = event.target.closest("[data-case-id]"); if (button) openCase(button.dataset.caseId).catch((error) => shell.toast(error.message)); };
    document.getElementById("case-detail-actions").onclick = (event) => { const button = event.target.closest("[data-action]"); if (button) runAction(button.dataset.action).catch((error) => shell.toast(error.message)); };
    await load();
    const requestedAssetId = new URLSearchParams(location.search).get("asset_id");
    if (requestedAssetId) await openCase(requestedAssetId);
  }).catch((error) => shell.toast(error.message));
})();
