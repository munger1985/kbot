(function () {
  "use strict";

  const api = "/api/v1/apps/aiops/operations-knowledge";
  const shell = globalThis.KBotAIOpsShell;
  let activeKind = "";
  let activeStatus = "";

  const splitValues = (value) => String(value || "").split(",").map((item) => item.trim()).filter(Boolean);
  const scope = (kind, value) => value ? [{ kind, value }] : [];

  function renderOverview(data) {
    document.querySelectorAll("[data-metric]").forEach((node) => {
      const key = node.dataset.metric;
      node.textContent = key === "abnormal"
        ? Number(data.failed || 0) + Number(data.index_abnormal || 0)
        : Number(data[key] || 0);
    });
  }

  function assetCard(item) {
    const kind = item.asset_kind === "MANUAL" ? "运维手册" : "诊断案例";
    return `<article class="knowledge-card">
      <header><div><h3>${shell.escape(item.display_name)}</h3><p>${kind}</p></div>${shell.badge(item.status)}</header>
      <div class="knowledge-card-meta"><span>安全等级 ${shell.escape(item.security_level)}</span><span>更新于 ${shell.escape(shell.fmt(item.updated_at))}</span></div>
      <footer><button type="button" data-open-version="${shell.escape(item.current_version_id || item.asset_id)}" data-asset-id="${shell.escape(item.asset_id)}">查看版本与审核</button></footer>
    </article>`;
  }

  async function loadAssets() {
    const query = new URLSearchParams({ limit: "200" });
    if (activeKind) query.set("asset_kind", activeKind);
    const result = await KBotAIOpsAuth.request(`${api}/assets?${query}`);
    let items = result.items || [];
    if (activeStatus) {
      const statuses = new Set(activeStatus.split(","));
      items = items.filter((item) => statuses.has(item.status));
    }
    document.getElementById("knowledge-assets").innerHTML = items.length
      ? items.map(assetCard).join("")
      : '<p class="ops-empty">当前筛选下没有知识资产。</p>';
  }

  function relationSection(title, body) {
    return `<section class="knowledge-section"><h3>${shell.escape(title)}</h3>${body}</section>`;
  }

  function textList(items, empty) {
    return items?.length
      ? `<ul>${items.map((item) => `<li>${shell.escape(typeof item === "string" ? item : item.summary || JSON.stringify(item))}</li>`).join("")}</ul>`
      : `<p class="ops-empty">${shell.escape(empty)}</p>`;
  }

  function renderManualProfile(profile) {
    const scopeItems = Object.entries(profile.scope || {}).flatMap(([kind, values]) =>
      (values || []).map((value) => `<span class="knowledge-scope">${shell.escape(kind)} · ${shell.escape(value)}</span>`));
    const procedures = (profile.procedures || []).map((procedure) => `<article class="knowledge-procedure">
      <h4>${shell.escape(procedure.title)}</h4>
      ${textList(procedure.preconditions, "未提取到前置条件")}
      ${(procedure.steps || []).map((step) => `<div class="knowledge-step"><strong>步骤 ${shell.escape(step.ordinal)}</strong><p>${shell.escape(step.description)}</p>${step.command_text ? `<pre class="ops-code"><code>${shell.escape(step.command_text)}</code></pre>` : ""}</div>`).join("") || '<p class="ops-empty">未提取到原文命令。</p>'}
      ${textList(procedure.validations, "未提取到验证步骤")}
      ${textList(procedure.rollback, "未提取到回退步骤")}
    </article>`).join("");
    return [
      profile.warnings?.length ? `<div class="knowledge-warning"><strong>需要人工确认</strong>${textList(profile.warnings, "")}</div>` : "",
      profile.missing_fields?.length ? `<p class="knowledge-status">缺失元数据：${shell.escape(profile.missing_fields.join("、"))}</p>` : "",
      `<div class="knowledge-scope-list">${scopeItems.join("") || '<span class="ops-empty">未识别适用范围</span>'}</div>`,
      procedures || '<p class="ops-empty">原文已索引，但没有识别出独立操作步骤。</p>',
    ].join("");
  }

  function renderCaseProfile(profile) {
    const signature = profile.problem_signature || {};
    const environment = profile.environment || {};
    return [
      `<dl class="ops-detail"><dt>案例等级</dt><dd>${shell.badge(profile.case_kind)}</dd><dt>问题分类</dt><dd>${shell.escape(signature.problem_class || "未分类")}</dd><dt>数据库环境</dt><dd>${shell.escape([environment.database_type, environment.database_version, environment.topology, environment.platform].filter(Boolean).join(" · ") || "未完整识别")}</dd></dl>`,
      `<h4>根因</h4><p>${shell.escape(profile.root_cause?.summary || "未形成根因")}</p>`,
      `<h4>处置动作</h4>${textList(profile.actions, "没有已执行动作，仅可作为诊断参考")}`,
      `<h4>验证</h4><p>${shell.escape(profile.verification?.result || "NOT_VERIFIED")}</p>`,
      `<h4>适用条件</h4>${textList(profile.applicability, "未补充额外适用条件")}`,
      `<h4>禁用条件</h4>${textList(profile.contraindications, "未记录禁用条件")}`,
    ].join("");
  }

  function renderProfile(profile, assetKind) {
    if (!profile) return '<p class="ops-empty">正文仍在解析和提炼中。</p>';
    return assetKind === "MANUAL" ? renderManualProfile(profile) : renderCaseProfile(profile);
  }

  async function openAsset(assetId) {
    const result = await KBotAIOpsAuth.request(`${api}/assets/${encodeURIComponent(assetId)}`);
    const version = result.versions?.[0];
    if (!version) throw new Error("该知识资产还没有版本");
    const detail = await KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(version.asset_version_id)}`);
    document.getElementById("detail-title").textContent = detail.asset.display_name;
    document.getElementById("detail-subtitle").textContent = `${detail.asset.asset_kind === "MANUAL" ? "运维手册" : "诊断案例"} · 版本 ${detail.version.version_no}`;
    const profile = detail.version.profile;
    document.getElementById("detail-body").innerHTML = [
      relationSection("状态", `<dl class="ops-detail"><dt>业务状态</dt><dd>${shell.badge(detail.version.status)}</dd><dt>索引状态</dt><dd>${detail.indexes.map((item) => shell.badge(item.status)).join(" ") || "—"}</dd><dt>来源摘要</dt><dd><code>${shell.escape(detail.version.source_hash)}</code></dd></dl>`),
      relationSection("适用范围", detail.scopes.length ? `<div class="knowledge-scope-list">${detail.scopes.map((item) => `<span class="knowledge-scope">${shell.escape(item.kind)} · ${shell.escape(item.value)}</span>`).join("")}</div>` : '<p class="ops-empty">尚未固化适用范围。</p>'),
      relationSection("提炼结果", renderProfile(profile, detail.asset.asset_kind)),
      relationSection("审核记录", detail.reviews.length ? detail.reviews.map((item) => `<div class="knowledge-review"><strong>${shell.escape(item.decision)}</strong> · ${shell.escape(item.reviewer_id)}<p>${shell.escape(item.comment || "未填写说明")} · ${shell.escape(shell.fmt(item.created_at))}</p></div>`).join("") : '<p class="ops-empty">尚无审核记录。</p>'),
    ].join("");
    const actions = [];
    actions.push('<button type="button" data-download-source>下载原始文件</button>');
    if (["PROCESSING", "FAILED"].includes(detail.version.status)) actions.push(`<button type="button" data-review="retry">重新处理</button>`);
    if (["DRAFT", "REVIEW_REQUIRED"].includes(detail.version.status)) {
      actions.push(`<button type="button" data-review="reject">拒绝</button>`);
      actions.push(`<button class="primary" type="button" data-review="publish">发布</button>`);
    }
    if (detail.version.status === "PUBLISHED") actions.push(`<button type="button" data-review="retire">退役</button>`);
    if (detail.asset.asset_kind === "MANUAL") actions.unshift('<button type="button" data-upload-version>上传新版本</button>');
    document.getElementById("detail-actions").innerHTML = actions.join("");
    document.getElementById("detail-actions").dataset.versionId = detail.version.asset_version_id;
    document.getElementById("detail-actions").dataset.rowVersion = detail.version.row_version;
    document.getElementById("detail-actions").dataset.assetId = detail.asset.asset_id;
    document.getElementById("detail-actions").dataset.assetRowVersion = detail.asset.row_version;
    document.getElementById("detail-actions").dataset.assetName = detail.asset.display_name;
    document.getElementById("detail-dialog").showModal();
  }

  async function review(event) {
    const uploadVersion = event.target.closest("[data-upload-version]");
    if (uploadVersion) {
      const footer = event.currentTarget;
      const form = document.getElementById("manual-form");
      form.dataset.assetId = footer.dataset.assetId;
      form.dataset.assetRowVersion = footer.dataset.assetRowVersion;
      document.getElementById("manual-name").value = footer.dataset.assetName || "";
      document.getElementById("upload-title").textContent = "上传手册新版本";
      document.getElementById("upload-subtitle").textContent = "新版本完成解析和审核前，当前已发布版本仍会继续参与检索。";
      document.getElementById("detail-dialog").close();
      document.getElementById("upload-dialog").showModal();
      return;
    }
    const download = event.target.closest("[data-download-source]");
    if (download) {
      const footer = event.currentTarget;
      await KBotAIOpsAuth.download(
        `${api}/versions/${encodeURIComponent(footer.dataset.versionId)}/source`,
        `operations-knowledge-${footer.dataset.versionId}`,
        "*/*",
      );
      return;
    }
    const button = event.target.closest("[data-review]");
    if (!button) return;
    const footer = event.currentTarget;
    const decision = button.dataset.review;
    button.disabled = true;
    try {
      const detail = await KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(footer.dataset.versionId)}:${decision}`, {
        method: "POST",
        body: JSON.stringify({ expected_row_version: Number(footer.dataset.rowVersion), comment: `通过运维知识库页面执行${decision}` }),
      });
      footer.dataset.rowVersion = detail.version.row_version;
      shell.toast("知识版本状态已更新");
      document.getElementById("detail-dialog").close();
      await refresh();
    } finally {
      button.disabled = false;
    }
  }

  async function uploadManual(event) {
    event.preventDefault();
    const file = document.getElementById("manual-file").files[0];
    if (!file) return;
    const button = event.currentTarget.querySelector('button[type="submit"]');
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", await file.arrayBuffer())))
      .map((value) => value.toString(16).padStart(2, "0")).join("");
    const scopes = [
      ...scope("DATABASE_TYPE", document.getElementById("manual-db").value),
      ...scope("DATABASE_VERSION", document.getElementById("manual-db-version").value),
      ...splitValues(document.getElementById("manual-topology").value).flatMap((value) => scope("TOPOLOGY", value)),
      ...splitValues(document.getElementById("manual-topics").value).flatMap((value) => scope("TOPIC", value)),
    ];
    const metadata = {
      display_name: document.getElementById("manual-name").value.trim() || file.name,
      publisher: document.getElementById("manual-publisher").value.trim() || null,
      document_version: document.getElementById("manual-version").value.trim() || null,
      notes: document.getElementById("manual-notes").value.trim() || null,
      security_level: 1,
      scopes,
    };
    button.disabled = true;
    document.getElementById("manual-result").textContent = "正在登记原文并进入解析队列…";
    try {
      const assetId = event.currentTarget.dataset.assetId;
      const endpoint = assetId
        ? `${api}/assets/${encodeURIComponent(assetId)}/versions`
        : `${api}/manuals`;
      const headers = {
        "Content-Type": file.type || "application/octet-stream",
        "Idempotency-Key": `aiops-manual:${assetId || "new"}:${digest}`,
        "X-File-Name": encodeURIComponent(file.name),
        "X-Content-SHA256": digest,
        "X-Upload-Metadata": encodeURIComponent(JSON.stringify(metadata)),
      };
      if (assetId) headers["X-Expected-Asset-Row-Version"] = event.currentTarget.dataset.assetRowVersion;
      await KBotAIOpsAuth.request(endpoint, {
        method: "POST",
        headers,
        body: file,
      });
      event.currentTarget.reset();
      delete event.currentTarget.dataset.assetId;
      delete event.currentTarget.dataset.assetRowVersion;
      document.getElementById("upload-title").textContent = "上传运维手册";
      document.getElementById("upload-subtitle").textContent = "用户填写值优先；与原文冲突时进入人工审核。";
      document.getElementById("upload-dialog").close();
      shell.toast("手册已受理，完成解析后需要人工发布");
      await refresh();
    } finally {
      button.disabled = false;
    }
  }

  async function search(event) {
    event.preventDefault();
    const node = document.getElementById("search-results");
    node.innerHTML = '<p class="ops-empty">正在筛选候选并检索原文…</p>';
    const result = await KBotAIOpsAuth.request(`${api}/search-preview`, {
      method: "POST",
      body: JSON.stringify({
        query: document.getElementById("search-query").value.trim(),
        purpose: "DIAGNOSE",
        database_type: document.getElementById("search-db").value || null,
        database_major_version: document.getElementById("search-version").value.trim() || null,
        topology: document.getElementById("search-topology").value.trim() || null,
        source_kinds: ["MANUAL", "DIAGNOSIS_CASE"], max_results: 8, max_security_level: 3,
      }),
    });
    node.innerHTML = result.results?.length
      ? result.results.map((item) => `<article class="knowledge-search-item"><h3>${shell.escape(item.title)} ${shell.badge(item.authority)}</h3><p>${shell.escape(item.asset_kind)} · ${shell.escape(item.applicability)} · ${item.citation_pack?.length || 0} 条原文引用</p></article>`).join("")
      : `<p class="ops-empty">${shell.escape(result.warnings?.join("；") || "没有找到适用于当前条件的已发布知识。")}</p>`;
  }

  async function loadReviews() {
    const result = await KBotAIOpsAuth.request(`${api}/reviews?limit=10`);
    document.getElementById("knowledge-reviews").innerHTML = result.items?.length
      ? result.items.map((item) => `<div class="knowledge-review"><strong>${shell.escape(item.decision)}</strong> · ${shell.escape(item.reviewer_id)}<p>${shell.escape(item.before_status)} → ${shell.escape(item.after_status)} · ${shell.escape(shell.fmt(item.created_at))}</p></div>`).join("")
      : '<p class="ops-empty">还没有审核记录。</p>';
  }

  async function refresh() {
    const [overview] = await Promise.all([
      KBotAIOpsAuth.request(`${api}/overview`), loadAssets(), loadReviews(),
    ]);
    renderOverview(overview);
  }

  shell.ready.then(async () => {
    document.querySelectorAll("[data-close]").forEach((button) => {
      button.onclick = () => button.closest("dialog").close();
    });
    document.getElementById("open-upload").onclick = () => {
      const form = document.getElementById("manual-form");
      form.reset();
      delete form.dataset.assetId;
      delete form.dataset.assetRowVersion;
      document.getElementById("upload-title").textContent = "上传运维手册";
      document.getElementById("upload-subtitle").textContent = "用户填写值优先；与原文冲突时进入人工审核。";
      document.getElementById("upload-dialog").showModal();
    };
    document.getElementById("manual-file").onchange = (event) => {
      if (!document.getElementById("manual-name").value) document.getElementById("manual-name").value = event.target.files[0]?.name || "";
    };
    document.getElementById("manual-form").addEventListener("submit", (event) => uploadManual(event).catch((error) => {
      document.getElementById("manual-result").textContent = error.message;
      shell.toast(error.message);
    }));
    document.getElementById("search-form").addEventListener("submit", (event) => search(event).catch((error) => shell.toast(error.message)));
    document.getElementById("knowledge-assets").onclick = (event) => {
      const button = event.target.closest("[data-asset-id]");
      if (button) openAsset(button.dataset.assetId).catch((error) => shell.toast(error.message));
    };
    document.getElementById("detail-actions").onclick = (event) => review(event).catch((error) => shell.toast(error.message));
    document.querySelectorAll(".knowledge-tabs button").forEach((button) => {
      button.onclick = async () => {
        document.querySelectorAll(".knowledge-tabs button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        activeKind = button.dataset.kind || "";
        activeStatus = button.dataset.status || "";
        await loadAssets();
      };
    });
    try { await refresh(); } catch (error) { shell.toast(error.message); }
  });
})();
