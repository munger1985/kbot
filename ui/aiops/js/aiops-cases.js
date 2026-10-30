(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/operations-knowledge";
  const shell = globalThis.KBotAIOpsShell;
  const state = { cases: [], detail: null, canManage: false, editing: false, feedbackId: null };
  const escape = (value) => shell.escape(value == null ? "" : value);
  const statusLabel = (value) => ({ PROCESSING: "提炼中", DRAFT: "待完善", REVIEW_REQUIRED: "待审核", PUBLISHED: "已发布", RETIRED: "已退役", REJECTED: "已拒绝", FAILED: "处理失败" }[value] || value || "—");
  const kindLabel = (value) => ({ DIAGNOSTIC_REFERENCE: "诊断参考", VERIFIED_RESOLUTION: "已验证解决方案", VERIFIED_OPTIMIZATION: "已验证优化", NEGATIVE_EXPERIENCE: "失败与无效经验" }[value] || "未分类");
  const badge = (value) => shell.badge(value, statusLabel(value));

  function renderOverview(summary) {
    const items = [["待完善", summary.draft], ["待审核", summary.review_required], ["已发布", summary.published], ["处理异常", summary.processing], ["已退役", summary.retired]];
    document.getElementById("case-overview").innerHTML = items.map(([label, value]) => `<article><span>${escape(label)}</span><strong>${escape(value || 0)}</strong></article>`).join("");
  }
  function renderRows() {
    const keyword = document.getElementById("case-search").value.trim().toLowerCase();
    const items = keyword ? state.cases.filter((item) => [item.display_name, item.root_cause_summary, item.problem_class, item.database_type, item.source_report_id].some((value) => String(value || "").toLowerCase().includes(keyword))) : state.cases;
    document.getElementById("case-count").textContent = `${items.length} 项`;
    document.getElementById("case-rows").innerHTML = items.length ? items.map((item) => `<tr><td><button class="table-primary-button" type="button" data-case-id="${escape(item.asset_id)}">${escape(item.display_name)}</button><small class="table-secondary">v${escape(item.version_no)} · L${escape(item.security_level)}</small></td><td>${escape(kindLabel(item.case_kind))}<small class="table-secondary">${escape(item.problem_class || "未分类")}</small></td><td>${escape([item.database_type, item.database_version, item.topology].filter(Boolean).join(" · ") || "通用")}</td><td class="technical-value">${escape(item.source_report_id || "—")}</td><td>${badge(item.status)}</td><td>${escape(shell.fmt(item.updated_at))}</td><td><button class="row-action" type="button" data-case-id="${escape(item.asset_id)}">查看</button></td></tr>`).join("") : '<tr><td colspan="7" class="empty-cell">当前条件下没有历史诊断案例。</td></tr>';
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
    document.getElementById("case-detail-body").innerHTML = `<form id="case-edit-form" class="drawer-form-section case-edit-form"><header><span>✎</span><div><strong>创建案例修订</strong><small>保存后生成新版本；来源报告、原始证据和验证引用不可修改。</small></div></header><label>案例名称<input name="display_name" maxlength="256" required value="${escape(detail.asset.display_name)}"></label><label>问题现象（每行一条）<textarea name="symptoms" rows="4" required>${escape(rows(signature.normalized_symptoms))}</textarea></label><div class="case-edit-grid"><label>组件<input name="component" value="${escape(signature.component)}"></label><label>数据库类型<input name="database_type" value="${escape(environment.database_type)}"></label><label>数据库版本<input name="database_version" value="${escape(environment.database_version)}"></label><label>平台<input name="platform" value="${escape(environment.platform)}"></label><label>拓扑<input name="topology" value="${escape(environment.topology)}"></label></div><label>调查结论与原因判断<textarea name="root_cause" rows="4" required>${escape(root.summary)}</textarea></label><label>已执行动作（每行一条）<textarea name="actions" rows="4">${escape(rows(profile.actions))}</textarea></label><label>适用条件（每行一条）<textarea name="applicability" rows="3">${escape(rows(profile.applicability))}</textarea></label><label>不适用条件（每行一条）<textarea name="contraindications" rows="3">${escape(rows(profile.contraindications))}</textarea></label></form>`;
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
    const withdrawals = (detail.external_withdrawals || []).map((item) => `<li><strong>${escape(({ PENDING: "等待知识服务确认", ACKNOWLEDGED: "知识服务已确认撤回", FAILED: "撤回未确认，需检查服务或来源" })[item.status] || "撤回状态未知")}</strong><small> · 已尝试 ${escape(item.attempt_count)} 次${item.receipt ? ` · ${escape(shell.fmt(item.receipt.withdrawn_at))}` : ` · ${escape(item.error_code || "")}`}</small></li>`).join("");
    document.getElementById("case-detail-eyebrow").textContent = `${kindLabel(profile.case_kind)} · v${detail.version.version_no}`;
    document.getElementById("case-detail-title").textContent = detail.asset.display_name;
    document.getElementById("case-detail-status").innerHTML = badge(detail.version.status);
    document.getElementById("case-detail-body").innerHTML = `<section class="drawer-form-section"><header><span>01</span><div><strong>问题现象与适用范围</strong><small>用于判断案例是否适合当前环境。</small></div></header><dl class="knowledge-summary-grid"><div><dt>问题分类</dt><dd>${escape(signature.problem_class || "未记录")}</dd></div><div><dt>组件</dt><dd>${escape(signature.component || "未记录")}</dd></div><div><dt>数据库</dt><dd>${escape([environment.database_type, environment.database_version].filter(Boolean).join(" ") || "未记录")}</dd></div><div><dt>拓扑</dt><dd>${escape(environment.topology || "未记录")}</dd></div><div><dt>症状</dt><dd>${escape(list(signature.normalized_symptoms).join("；") || "未记录")}</dd></div></dl></section><section class="drawer-form-section"><header><span>02</span><div><strong>调查结论与原因判断</strong><small>诊断参考保留已知事实、原因强度与限制，发布仍需来源核验和人工审核。</small></div></header><p>${escape(root.summary || "未记录根因")}</p></section><section class="drawer-form-section"><header><span>03</span><div><strong>处置与验证</strong><small>人工登记恢复不等于受控验证；不同前后引用不能自动证明措施有效。</small></div></header><ol>${actions.map((item) => `<li>${escape(item.summary)}</li>`).join("") || "<li>未记录已执行动作，仅作为诊断参考。</li>"}</ol><p>验证结果：${escape(verification.result || "NOT_VERIFIED")}</p></section><section class="drawer-form-section"><header><span>04</span><div><strong>来源追溯</strong><small>保留正式报告、运行和证据引用。</small></div></header><div class="knowledge-source-list">${sourceRows}</div>${reportId ? `<a class="ops-button" href="./report-detail.html?id=${encodeURIComponent(reportId)}">查看来源报告</a>` : ""}</section><section class="drawer-form-section"><header><span>05</span><div><strong>版本与审核记录</strong><small>发布、拒绝、来源更正和退役均保留审计。</small></div></header><ol class="event-timeline knowledge-review-timeline">${reviews}</ol></section>${withdrawals ? `<section class="drawer-form-section" aria-label="外部知识撤回"><header><span>↩</span><div><strong>外部知识撤回</strong><small>仅代表 Knowledge Core 已提交撤回，不代表未登记消费者的缓存已清除。</small></div></header><ul>${withdrawals}</ul><p>该版本不能原地重新发布。请重新取证并提炼案例，或拒绝当前版本后创建修订；新版本仍需核验来源和审核。</p></section>` : ""}`;
    const actionsHtml = [];
    if (state.canManage && ["DRAFT", "PUBLISHED", "REJECTED", "RETIRED"].includes(detail.version.status)) actionsHtml.push('<button type="button" data-action="edit">编辑并创建新版本</button>');
    if (state.canManage && detail.version.status === "PROCESSING") actionsHtml.push('<button type="button" data-action="reconcile">立即处理</button>');
    if (state.canManage && detail.version.status === "DRAFT") actionsHtml.push('<button class="primary" type="button" data-action="submit-review">提交审核</button>');
    if (state.canManage && detail.version.status === "REVIEW_REQUIRED") actionsHtml.push('<button class="danger" type="button" data-action="reject">拒绝</button>');
    if (state.canManage && detail.version.status === "REVIEW_REQUIRED" && !(detail.external_withdrawals || []).length) actionsHtml.push('<button class="primary" type="button" data-action="publish">审核通过并发布</button>');
    if (state.canManage && detail.version.status === "PUBLISHED") actionsHtml.push('<button class="danger" type="button" data-action="retire">退役</button>');
    if (state.canManage && detail.version.status === "FAILED") actionsHtml.push('<button class="primary" type="button" data-action="retry">重试处理</button>');
    document.getElementById("case-detail-actions").innerHTML = actionsHtml.join("");
    document.getElementById("case-detail-body").insertAdjacentHTML("beforeend", `<form id="case-feedback-form" class="drawer-form-section case-edit-form"><header><span>06</span><div><strong>登记复用反馈</strong><small>人工报告单独留痕，不改变验证等级或发布状态。</small></div></header><label>关联本人的诊断 Run（可选）<input name="source_run_id" maxlength="36"></label><label>环境适用性<select name="applicability"><option value="UNCERTAIN">尚未核实</option><option value="APPLICABLE">适用</option><option value="NOT_APPLICABLE">不适用</option></select></label><label>检查路径价值<select name="check_value"><option value="UNKNOWN">尚未判断</option><option value="USEFUL">有帮助</option><option value="NOT_USEFUL">无帮助</option></select></label><label>报告的措施效果<select name="action_effect"><option value="NOT_ATTEMPTED">尚未实施</option><option value="IMPROVED">有改善</option><option value="UNCHANGED">无变化</option><option value="DEGRADED">出现恶化</option><option value="INCONCLUSIVE">无法判断</option></select></label><label>差异、检查结果或失败原因<textarea name="notes" required maxlength="1000" rows="3"></textarea></label><button type="submit">保存人工反馈</button></form>`);
  }
  async function openCase(assetId) {
    state.feedbackId = KBotAIOpsAuth.uuid7();
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
    if (action === "publish" && !confirm("确认已核对脱敏预览、来源有效性、适用条件及失败风险，再发布供 Agent 检索？自动脱敏不保证识别全部敏感信息。")) return;
    const options = action === "reconcile" ? { method: "POST" } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ expected_row_version: Number(version.row_version), comment: null }) };
    await KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(version.asset_version_id)}:${action}`, options);
    shell.toast("案例状态已更新");
    document.getElementById("case-drawer").close();
    await load();
  }
  shell.ready.then(async (access) => {
    document.getElementById("case-detail-body").addEventListener("submit", async (event) => {
      if (event.target.id !== "case-feedback-form" || !state.detail) return;
      event.preventDefault();
      const form = event.target;
      const submit = form.querySelector('[type="submit"]');
      if (submit.disabled || !form.reportValidity()) return;
      submit.disabled = true;
      const data = new FormData(form);
      try {
        await KBotAIOpsAuth.request(`${api}/cases/${encodeURIComponent(state.detail.asset.asset_id)}/feedback`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ feedback_id: state.feedbackId, asset_version_id: state.detail.version.asset_version_id,
            source_run_id: String(data.get("source_run_id") || "").trim() || null,
            applicability: data.get("applicability"), check_value: data.get("check_value"),
            action_effect: data.get("action_effect"), notes: String(data.get("notes") || "").trim() }),
        });
        shell.toast("复用反馈已留痕，不改变案例验证等级或发布状态");
        await openCase(state.detail.asset.asset_id);
      } catch (error) { shell.toast(error.message); }
      finally { submit.disabled = false; }
    });
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
