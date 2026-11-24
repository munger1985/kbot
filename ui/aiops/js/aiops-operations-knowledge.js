(function () {
  "use strict";

  const api = "/api/v1/apps/aiops/operations-knowledge";
  const shell = globalThis.KBotAIOpsShell;
  const state = {
    assets: [],
    detail: null,
    detailTab: "overview",
    entries: [],
    uploadAsset: null,
    tags: [],
    selectedTagIds: [],
    knowledgeBases: [],
    uploading: false,
    submittingReview: false,
  };

  const isProcessing = (status) => ["TAGGING", "PROCESSING"].includes(status);
  let assetRequest = 0;
  let loadingAssets = false;
  const escape = (value) => shell.escape(value == null ? "" : value);
  const newId = () => KBotAIOpsAuth.uuid();
  const kindLabel = (value) => value === "MANUAL" ? "运维手册" : "诊断案例";
  const tagCategories = [["RESOURCE_CATEGORY", "资源分类"], ["TECHNOLOGY_STACK", "技术栈"], ["VENDOR", "厂商"]];
  const statusLabel = (value) => ({ TAGGING: "标签识别中", TAGGING_FAILED: "标签识别失败", PROCESSING: "解析中", DRAFT: "待发布审核", REVIEW_REQUIRED: "待审批", PUBLISHED: "已发布", RETIRED: "已退役", REJECTED: "已拒绝", FAILED: "解析失败" }[value] || value || "—");
  const statusTone = (value) => ["FAILED", "TAGGING_FAILED", "REJECTED"].includes(value) ? "bad" : ["TAGGING", "PROCESSING", "DRAFT", "REVIEW_REQUIRED"].includes(value) ? "warn" : ["PUBLISHED", "READY"].includes(value) ? "good" : "";
  const badge = (value, label = statusLabel(value)) => `<span class="status-pill ${({ good: "status-success", bad: "status-danger", warn: "status-warning" })[statusTone(value)] || "status-muted"}">${escape(label)}</span>`;
  const reviewLabel = (value) => value === "REVIEW_REQUIRED" ? "待审批" : ["TAGGING", "TAGGING_FAILED", "PROCESSING", "PUBLISHED", "RETIRED"].includes(value) ? "已通过" : value === "REJECTED" ? "已拒绝" : value === "DRAFT" ? "待审核" : "—";
  const parseLabel = (value, kind = "MANUAL") => value === "TAGGING" ? "AI 识别中" : value === "TAGGING_FAILED" ? "AI 识别失败" : value === "REVIEW_REQUIRED" || (kind === "MANUAL" && ["DRAFT", "REJECTED"].includes(value)) ? "未开始" : value === "PROCESSING" ? "解析中" : value === "FAILED" ? "解析失败" : ["PUBLISHED", "RETIRED"].includes(value) ? "解析完成" : "—";
  const formatBytes = (bytes) => bytes < 1024 ? `${bytes} B` : bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(bytes < 10240 ? 1 : 0)} KB` : `${(bytes / 1024 ** 2).toFixed(bytes < 10 * 1024 ** 2 ? 1 : 0)} MB`;
  const pendingUploads = () => state.entries.filter(entry => !entry.accepted);
  const totalBytes = () => pendingUploads().reduce((sum, item) => sum + item.file.size, 0);
  const uploadLocked = () => state.entries.some(entry => entry.started);
  const selectedKnowledgeBaseId = () => document.getElementById("knowledge-base-filter").value;

  const digest = (file) => KBotSha256.file(file);

  function assetRow(item) {
    const action = item.status === "REVIEW_REQUIRED" ? "审批资料" : ["TAGGING", "PROCESSING"].includes(item.status) ? "查看进度" : ["FAILED", "TAGGING_FAILED"].includes(item.status) ? "处理异常" : item.status === "DRAFT" ? "审核" : "查看";
    return `<tr><td><button class="table-primary-button" type="button" data-asset-id="${escape(item.asset_id)}">${escape(item.display_name)}</button><small class="table-secondary">${escape(kindLabel(item.asset_kind))} · L${escape(item.security_level)} · ${escape(item.asset_id)}</small></td><td>${escape(state.knowledgeBases.find(base => base.knowledge_base_id === item.knowledge_base_id)?.display_name || "知识库已不可用")}</td><td>${badge(item.status, reviewLabel(item.status))}</td><td>${badge(item.status, parseLabel(item.status, item.asset_kind))}</td><td><span class="knowledge-submitted-time">${escape(shell.fmt(item.created_at))}</span>${item.updated_at !== item.created_at ? `<small class="table-secondary">更新于 ${escape(shell.fmt(item.updated_at))}</small>` : ""}</td><td><button class="row-action" type="button" data-asset-id="${escape(item.asset_id)}">${action}</button></td></tr>`;
  }

  function renderAssets() {
    const keyword = document.getElementById("knowledge-search").value.trim().toLowerCase();
    const items = keyword ? state.assets.filter((item) => [item.display_name, item.asset_id, kindLabel(item.asset_kind)].some((value) => String(value).toLowerCase().includes(keyword))) : state.assets;
    document.getElementById("knowledge-count").textContent = `${items.length} 项`;
    document.getElementById("knowledge-assets").innerHTML = items.length ? items.map(assetRow).join("") : '<tr><td colspan="6" class="empty-cell">当前条件下没有运维资料。</td></tr>';
  }

  async function loadAssets() {
    const knowledgeBaseId = selectedKnowledgeBaseId();
    const query = new URLSearchParams({ limit: "200", asset_kind: "MANUAL" });
    if (knowledgeBaseId) query.set("knowledge_base_id", knowledgeBaseId);
    const status = document.getElementById("knowledge-status").value;
    if (status) query.set("status", status);
    const requestId = ++assetRequest;
    loadingAssets = true;
    try {
      const result = await KBotAIOpsAuth.request(`${api}/assets?${query}`);
      if (requestId !== assetRequest) return;
      state.assets = result.items || [];
      renderAssets();
    } finally {
      if (requestId === assetRequest) loadingAssets = false;
    }
  }

  async function loadKnowledgeBases() {
    const result = await KBotAIOpsAuth.request(`${api}/knowledge-bases`);
    state.knowledgeBases = result.items || [];
    const preferred = new URLSearchParams(location.search).get("knowledge_base_id");
    const select = document.getElementById("knowledge-base-filter");
    select.innerHTML = '<option value="">全部知识库</option>' + (state.knowledgeBases.length
      ? state.knowledgeBases.map((item) => `<option value="${escape(item.knowledge_base_id)}">${escape(item.display_name)}</option>`).join("")
      : "");
    if (preferred && state.knowledgeBases.some((item) => item.knowledge_base_id === preferred)) select.value = preferred;
    const activeBases = state.knowledgeBases.filter(item => item.status === 'ACTIVE');
    document.getElementById("upload-knowledge-base").innerHTML = '<option value="" disabled>请选择</option>' + activeBases.map(item => `<option value="${escape(item.knowledge_base_id)}">${escape(item.display_name)}</option>`).join("");
    document.getElementById("upload-no-knowledge-base").hidden = Boolean(activeBases.length);
    document.getElementById("open-upload").disabled = false;
  }

  async function loadTags() {
    const result = await KBotAIOpsAuth.request(`${api}/tags?include_disabled=true`);
    state.tags = result.items || [];
  }

  function renderUploadTaggingHint() {
    const id = state.uploadAsset?.knowledge_base_id || document.getElementById("upload-knowledge-base").value;
    const tagging = state.knowledgeBases.find(item => item.knowledge_base_id === id)?.tagging;
    const hint = document.getElementById("upload-tagging-hint");
    if (!hint) return;
    const missing = [];
    if (tagging?.state === "NOT_CONFIGURED") missing.push("标签识别模型");
    if (tagging?.state === "DEGRADED") missing.push("可用的标签识别能力");
    if (tagging?.vocabulary_ready === false) missing.push("有效标签词表");
    hint.hidden = !missing.length;
    hint.textContent = missing.length ? "可以上传待审资料；审批开始标签处理前需要补齐：" + missing.join("、") + "。" : "";
  }

  function stageClass(stage, status, kind = "MANUAL") {
    if (stage === "uploaded") return "is-complete";
    if (stage === "approved") return ["DRAFT", "REVIEW_REQUIRED"].includes(status) ? "is-current" : status === "REJECTED" ? "is-error" : ["TAGGING", "TAGGING_FAILED", "PROCESSING", "FAILED", "PUBLISHED", "RETIRED"].includes(status) ? "is-complete" : "";
    if (stage === "tagged") return status === "TAGGING" ? "is-current" : status === "TAGGING_FAILED" ? "is-error" : ["PROCESSING", "FAILED", "PUBLISHED", "RETIRED"].includes(status) ? "is-complete" : "";
    if (stage === "parsed") return status === "PROCESSING" ? "is-current" : status === "FAILED" ? "is-error" : kind === "DIAGNOSIS_CASE" && ["DRAFT", "REVIEW_REQUIRED"].includes(status) ? "is-complete" : ["PUBLISHED", "RETIRED"].includes(status) ? "is-complete" : "";
    return ["PUBLISHED", "RETIRED"].includes(status) ? "is-complete" : "";
  }

  function tagCatalog(detail) {
    const catalog = new Map(state.tags.map((item) => [item.tag_id, item]));
    for (const item of detail.version.tag_analysis?.vocabulary || []) {
      if (!catalog.has(item.tag_id)) catalog.set(item.tag_id, { ...item, status: "DISABLED" });
    }
    return [...catalog.values()];
  }

  function renderTags(detail) {
    if (detail.asset.asset_kind !== "MANUAL") return "";
    const analysis = detail.version.tag_analysis || {};
    const suggestions = new Map((analysis.suggestions || []).map((item) => [item.tag_id, item]));
    const catalog = tagCatalog(detail);
    const groups = tagCategories.map(([category, label]) => {
      const tags = catalog.filter((item) => item.category === category && (item.status === "ACTIVE" || state.selectedTagIds.includes(item.tag_id) || suggestions.has(item.tag_id)));
      const options = tags.length ? tags.map((tag) => {
        const suggestion = suggestions.get(tag.tag_id);
        const evidence = (suggestion?.evidence || []).map((item) => `<article class="knowledge-tag-evidence"><small>${escape(item.file_name || "原始文件")}</small><q>${escape(item.quote)}</q></article>`).join("");
        return `<label class="knowledge-tag-option${suggestion ? " suggested" : ""}"><input type="checkbox" ${state.selectedTagIds.includes(tag.tag_id) ? "checked" : ""} disabled><span><strong>${escape(tag.display_name)}</strong><small>${escape(tag.description || "暂无说明")}</small>${suggestion ? `<em>AI 识别 · ${Math.round(Number(suggestion.confidence || 0) * 100)}%</em>` : ""}</span></label>${evidence}`;
      }).join("") : '<p class="ops-empty">该分类尚未配置有效标签。</p>';
      return `<fieldset class="knowledge-tag-group"><legend>${escape(label)}</legend>${options}</fieldset>`;
    }).join("");
    const failure = analysis.status === "FAILED" ? `<div class="knowledge-warning knowledge-error"><strong>标签识别失败</strong><p>${escape(analysis.error_summary || analysis.error_code || "未知错误")}</p></div>` : "";
    const warnings = analysis.warnings?.length ? `<div class="knowledge-warning"><strong>识别提示</strong><p>${escape(analysis.warnings.join("；"))}</p></div>` : "";
    const hint = detail.version.status === "REVIEW_REQUIRED" ? "资料审批通过后才会调用 AI 识别标签。" : "展示 AI 根据原文证据自动冻结的标签。";
    return `<section class="drawer-form-section knowledge-tag-review"><header><span>02</span><div><strong>受控标签</strong><small>${hint}</small></div></header>${failure}${groups}${warnings}</section>`;
  }

  function renderOverview(detail) {
    const status = detail.version.status;
    const scope = detail.scopes.map((item) => `${item.kind}=${item.value}`).join("；") || "通用";
    const stages = detail.asset.asset_kind === "MANUAL"
      ? `<section class="knowledge-stage-track"><div class="${stageClass("uploaded", status)}"><span>1</span><strong>上传</strong></div><div class="${stageClass("approved", status)}"><span>2</span><strong>审批</strong></div><div class="${stageClass("tagged", status)}"><span>3</span><strong>标签识别</strong></div><div class="${stageClass("parsed", status)}"><span>4</span><strong>文档解析</strong></div><div class="${stageClass("published", status)}"><span>5</span><strong>发布</strong></div></section>`
      : `<section class="knowledge-stage-track four-steps"><div class="${stageClass("uploaded", status, "DIAGNOSIS_CASE")}"><span>1</span><strong>生成</strong></div><div class="${stageClass("parsed", status, "DIAGNOSIS_CASE")}"><span>2</span><strong>文档解析</strong></div><div class="${stageClass("approved", status, "DIAGNOSIS_CASE")}"><span>3</span><strong>审批</strong></div><div class="${stageClass("published", status, "DIAGNOSIS_CASE")}"><span>4</span><strong>发布</strong></div></section>`;
    return `${stages}<section class="drawer-form-section"><header><span>01</span><div><strong>资料概览</strong><small>当前资料及其最新知识版本。</small></div></header><dl class="knowledge-summary-grid"><div><dt>资料类型</dt><dd>${escape(kindLabel(detail.asset.asset_kind))}</dd></div><div><dt>安全级</dt><dd>L${escape(detail.asset.security_level)}</dd></div><div><dt>当前版本</dt><dd>v${escape(detail.version.version_no)}</dd></div><div><dt>受控标签</dt><dd>${escape(scope)}</dd></div><div><dt>提交时间</dt><dd>${escape(shell.fmt(detail.asset.created_at))}</dd></div><div><dt>最近更新</dt><dd>${escape(shell.fmt(detail.asset.updated_at))}</dd></div></dl></section>${renderTags(detail)}${detail.version.extraction_warnings?.length ? `<section class="knowledge-warning"><strong>解析提示</strong><ul>${detail.version.extraction_warnings.map((item) => `<li>${escape(item)}</li>`).join("")}</ul></section>` : ""}`;
  }

  function renderFiles(detail) {
    const files = (detail.sources || []).filter(source => source.source_kind === "MANUAL_FILE");
    const sources = files.length ? files.map(source => `<article class="knowledge-source-card"><div><strong>${escape(source.external_id || detail.asset.display_name)}</strong><code>${escape(source.content_hash)}</code></div></article>`).join("") : '<p class="ops-empty">尚未登记上传原文。</p>';
    const indexes = detail.indexes?.length ? detail.indexes.map((item) => `<tr><td><code>${escape(item.index_ref_id)}</code></td><td>${badge(item.status, item.status)}</td><td>${escape(item.error_summary || "—")}</td></tr>`).join("") : '<tr><td colspan="3" class="ops-empty">尚未登记索引结果。</td></tr>';
    return `<section class="drawer-form-section"><header><span>01</span><div><strong>原始文件</strong><small>查看上传原文，可下载核验。</small></div></header><div class="knowledge-source-list">${sources}</div><div class="knowledge-source-actions">${(detail.indexes || []).map((item, index) => `<button type="button" data-download-source="${escape(item.document_version_id)}">下载文件 ${index + 1}</button>`).join("")}</div><dl class="knowledge-technical-list"><dt>资料内容指纹</dt><dd><code>${escape(detail.version.source_hash)}</code></dd></dl></section><section class="drawer-form-section"><header><span>02</span><div><strong>索引状态</strong><small>用于判断当前版本是否已经形成可检索知识。</small></div></header><div class="ops-table-wrap"><table class="ops-table"><thead><tr><th>索引</th><th>状态</th><th>异常</th></tr></thead><tbody>${indexes}</tbody></table></div></section>${detail.version.profile ? `<section class="drawer-form-section"><header><span>03</span><div><strong>提炼结果</strong><small>结构化结果仅辅助导航，运维结论仍应以原文为准。</small></div></header><pre class="knowledge-profile">${escape(JSON.stringify(detail.version.profile, null, 2))}</pre></section>` : ""}`;
  }

  function renderHistory(detail) {
    const rows = detail.reviews?.length ? detail.reviews.map((item) => `<li><div><strong>${escape(item.decision)}</strong><small>${escape(item.reviewer_id)}</small></div><time>${escape(shell.fmt(item.created_at))}</time></li>`).join("") : '<p class="ops-empty">当前版本还没有审核记录。</p>';
    return `<section class="drawer-form-section"><header><span>01</span><div><strong>处理记录</strong><small>保留当前版本的审核决策与责任人。</small></div></header><ol class="knowledge-timeline">${rows}</ol></section>`;
  }

  function renderDetailBody() {
    if (!state.detail) return;
    document.getElementById("detail-body").innerHTML = state.detailTab === "overview" ? renderOverview(state.detail) : state.detailTab === "files" ? renderFiles(state.detail) : renderHistory(state.detail);
    document.querySelectorAll("#detail-tabs [data-tab]").forEach((button) => button.classList.toggle("active", button.dataset.tab === state.detailTab));
  }

  function renderDetailActions(detail) {
    const actions = [];
    if (detail.asset.asset_kind === "MANUAL") actions.push('<button class="secondary-button" type="button" data-upload-version>上传新版本</button>');
    if (detail.version.status === "PROCESSING") actions.push('<button class="secondary-button" type="button" data-review="reconcile">立即处理</button>');
    if (detail.asset.asset_kind === "MANUAL" && ["DRAFT", "REVIEW_REQUIRED", "TAGGING", "TAGGING_FAILED", "PROCESSING", "FAILED", "REJECTED"].includes(detail.version.status)) actions.push('<button class="danger-button" type="button" data-delete-asset>删除资料</button>');
    if (["DRAFT", "REVIEW_REQUIRED", "TAGGING_FAILED"].includes(detail.version.status)) actions.push('<button class="danger-button" type="button" data-review="reject">拒绝</button>');
    if (detail.asset.asset_kind === "MANUAL" && detail.version.status === "REVIEW_REQUIRED") {
      const labels = { TEXT: "原生解析", AUTO: "自动判断", VISUAL: "视觉增强" };
      const options = (detail.parse_strategy_options || []).map(option => {
        const reason = option.executor_unavailable ? "模型解析执行端未启用" : (option.unavailable_roles || []).map(role => role.toUpperCase()).join("、") + " 不可用";
        return `<option value="${escape(option.strategy)}"${option.selectable ? "" : " disabled"}>${escape(labels[option.strategy] + (option.selectable ? "" : `（${reason}）`))}</option>`;
      });
      actions.unshift(`<label class="control-field knowledge-detail-strategy"><span class="field-label">批准时选择解析方案</span><select id="manual-parse-strategy"><option value="" disabled selected>选择解析方案</option>${options.join("")}</select></label>`);
      actions.push('<button class="primary-button" type="button" data-review="approve" disabled>批准并开始 AI 识别</button>');
    }
    if (detail.version.status === "PUBLISHED") actions.push('<button class="danger-button" type="button" data-review="retire">退役</button>');
    if (["FAILED", "TAGGING_FAILED"].includes(detail.version.status)) actions.push(`<button class="primary-button" type="button" data-review="retry">${detail.version.status === "TAGGING_FAILED" ? "重试标签识别" : "重试解析"}</button>`);
    document.getElementById("detail-actions").innerHTML = actions.join("");
    const strategy = document.getElementById("manual-parse-strategy");
    if (strategy) {
      strategy.value = detail.parse_strategy_options?.some(option => option.strategy === "AUTO" && option.selectable) ? "AUTO" : "";
      const approve = document.querySelector('#detail-actions [data-review="approve"]');
      const syncApproval = () => { approve.disabled = !strategy.value; };
      strategy.onchange = syncApproval;
      syncApproval();
    }
  }

  async function openAsset(assetId) {
    const asset = await KBotAIOpsAuth.request(`${api}/assets/${encodeURIComponent(assetId)}`);
    const version = asset.versions?.[0];
    if (!version) throw new Error("该资料尚未形成知识版本");
    state.detail = await KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(version.asset_version_id)}`);
    const assigned = state.detail.version.tag_analysis?.assigned_tag_ids || state.detail.scopes.map((item) => item.tag_id).filter(Boolean);
    state.selectedTagIds = assigned.length ? assigned : (state.detail.version.tag_analysis?.suggestions || []).map((item) => item.tag_id);
    state.detailTab = "overview";
    document.getElementById("detail-eyebrow").textContent = `${kindLabel(state.detail.asset.asset_kind)} · v${state.detail.version.version_no}`;
    const title = document.getElementById("detail-title");
    title.textContent = title.title = state.detail.asset.display_name;
    document.getElementById("detail-subtitle").textContent = state.detail.asset.asset_id;
    document.getElementById("detail-status").innerHTML = badge(state.detail.version.status);
    renderDetailBody();
    renderDetailActions(state.detail);
    document.getElementById("detail-drawer").showModal();
    document.getElementById("detail-body").scrollTop = 0;
  }

  function renderUploadLedger() {
    const ledger = document.getElementById("upload-ledger");
    ledger.hidden = !state.entries.length;
    const pending = pendingUploads();
    const disabled = state.uploading ? " disabled" : "";
    document.getElementById("upload-total").textContent = `${state.entries.length} 个 · 已接收 ${state.entries.length - pending.length} 个 · 待提交 ${formatBytes(totalBytes())}`;
    document.getElementById("upload-files").innerHTML = state.entries.map((entry, index) =>
      `<article class="upload-file-row" data-entry-id="${escape(entry.id)}"><span>${index + 1}</span><div><strong>${escape(entry.file.name)}</strong><small>${formatBytes(entry.file.size)}</small></div>
      ${state.uploadAsset ? "" : `<label class="control-field"><span>资料名称</span><input data-file-name value="${escape(entry.name)}" maxlength="256"${entry.started ? " disabled" : disabled}></label>`}
      <span class="status-pill">${escape(entry.accepted ? "已接收，等待审批" : entry.error || "待提交")}</span><button type="button" data-remove-file${disabled}>移除</button></article>`).join("");
    document.getElementById("clear-files").disabled = state.uploading;
    document.getElementById("choose-files").disabled = state.uploading || Boolean(state.uploadAsset && uploadLocked());
    document.getElementById("upload-knowledge-base").disabled = state.uploading || Boolean(state.uploadAsset) || uploadLocked();
    document.getElementById("manual-security").disabled = state.uploading || uploadLocked();
    document.getElementById("manual-name").disabled = !state.uploadAsset || state.uploading || uploadLocked();
    updateSubmitUpload();
  }

  function updateSubmitUpload() {
    const pending = pendingUploads();
    document.getElementById("submit-upload").disabled = state.uploading || !pending.length ||
      (state.uploadAsset ? !document.getElementById("manual-name").value.trim() :
        !document.getElementById("upload-knowledge-base").value || pending.some(entry => !entry.name.trim())) ||
      totalBytes() > 64 * 1024 * 1024 || state.entries.length > 20;
  }

  function addFiles(files) {
    if (state.uploading || !files.length || (state.uploadAsset && uploadLocked())) return;
    const entries = (state.uploadAsset ? files.slice(0, 1) : files).map(file => ({
      id: newId(), file, name: (file.name.replace(/\.[^.]+$/, "") || file.name).trim().slice(0, 256) || "未命名文件"
    }));
    if (state.uploadAsset) state.entries = entries;
    else state.entries.push(...entries);
    document.getElementById("manual-result").textContent = state.entries.length > 20 || totalBytes() > 64 * 1024 * 1024 ? "最多 20 个文件，总计不超过 64 MiB。" : "";
    renderUploadLedger();
  }

  function resetUpload(asset = null) {
    state.uploadAsset = asset;
    state.entries = [];
    document.getElementById("manual-form").reset();
    document.getElementById("manual-result").textContent = "";
    document.getElementById("manual-name").value = asset?.display_name || "";
    document.getElementById("manual-security").value = String(asset?.security_level || 1);
    document.getElementById("upload-title").textContent = asset ? "上传新版本" : "上传资料";
    document.getElementById("upload-subtitle").textContent = asset ? "新版本上传后先等待人工审批；批准后才调用 AI 识别标签并进入文档解析，当前发布版本继续参与检索。" : "上传后先进入人工审批；批准后才调用 AI 识别标签并解析。可选择一个或多个文件，每个文件独立提交和审核。";
    document.getElementById("upload-knowledge-base-section").hidden = Boolean(asset);
    const activeBases = state.knowledgeBases.filter(item => item.status === 'ACTIVE');
    document.getElementById("upload-knowledge-base").value = asset?.knowledge_base_id || activeBases.find(item => item.knowledge_base_id === selectedKnowledgeBaseId())?.knowledge_base_id || activeBases[0]?.knowledge_base_id || "";
    document.getElementById("upload-knowledge-base").disabled = Boolean(asset);
    renderUploadTaggingHint();
    document.getElementById("file-step").textContent = asset ? "01" : "02";
    document.getElementById("metadata-step").textContent = asset ? "02" : "03";
    document.getElementById("manual-file").multiple = !asset;
    document.getElementById("manual-name-field").hidden = !asset;
    document.getElementById("drop-help").textContent = asset ? "上传新版本只选择一个文件。" : "可多次选择并追加文件，最多 20 个、总计 64 MiB。";
    renderUploadLedger();
  }

  async function submitUpload(event) {
    event.preventDefault();
    if (document.getElementById("submit-upload").disabled) return;
    const base = document.getElementById("upload-knowledge-base").value;
    const security_level = Number(document.getElementById("manual-security").value);
    const versionTitle = document.getElementById("manual-name").value.trim();
    state.uploading = true;
    document.querySelectorAll('#upload-drawer [data-close]').forEach(control => { control.disabled = true; });
    document.getElementById("manual-result").textContent = "正在上传…";
    renderUploadLedger();
    let accepted = 0;
    try {
      if (state.uploadAsset) {
        const entry = pendingUploads()[0];
        entry.started = true;
        const { id, file } = entry;
        const metadata = { display_name: versionTitle, security_level };
        await KBotAIOpsAuth.request(`${api}/assets/${encodeURIComponent(state.uploadAsset.asset_id)}/versions`, { method: "POST",
          headers: { "Content-Type": file.type || "application/octet-stream", "Idempotency-Key": id, "X-File-Name": encodeURIComponent(file.name),
            "X-Content-SHA256": await digest(file), "X-Upload-Metadata": encodeURIComponent(JSON.stringify(metadata)), "X-Expected-Asset-Row-Version": String(state.uploadAsset.row_version) }, body: file });
        accepted = 1;
      } else {
        for (const entry of pendingUploads()) {
          entry.started = true;
          try {
            const { id, file } = entry, display_name = entry.name.trim();
            const body = new FormData();
            body.set("knowledge_base_id", base);
            body.set("metadata", JSON.stringify({ display_name, security_level }));
            body.set("files", JSON.stringify([{ part_name: "file", client_file_id: id, display_name,
              declared_mime_type: file.type || "application/octet-stream", byte_size: file.size, content_sha256: await digest(file) }]));
            body.append("file", file);
            const receipt = await KBotAIOpsAuth.request(`${api}/uploads`, { method: "POST", headers: { "Idempotency-Key": id }, body });
            const result = receipt.items.find(item => item.client_file_id === id);
            if (result?.status !== "ACCEPTED") throw new Error(result?.message || "未确认接收，可使用原提交标识重试。");
            entry.accepted = true; entry.error = ""; accepted++;
          } catch (error) {
            entry.error = error?.message || "未确认接收，可使用原提交标识重试。";
          }
        }
      }
      document.getElementById("manual-result").textContent = pendingUploads().length && !state.uploadAsset ? "未接收的文件可单独重试。" : "";
      if (state.uploadAsset || !pendingUploads().length) document.getElementById("upload-drawer").close();
      if (accepted) { shell.toast(`已接收 ${accepted} 个文件，等待人工审批`); await loadAssets(); }
    } finally {
      state.uploading = false;
      document.querySelectorAll('#upload-drawer [data-close]').forEach(control => { control.disabled = false; });
      renderUploadLedger();
    }
  }

  async function downloadSource(documentVersionId) {
    if (!state.detail) return;
    await KBotAIOpsAuth.download(`${api}/versions/${encodeURIComponent(state.detail.version.asset_version_id)}/source${documentVersionId ? `?document_version_id=${encodeURIComponent(documentVersionId)}` : ""}`, state.detail.asset.display_name, "*/*");
  }

  async function confirmAssetDeletion(displayName) {
    const result = await shell.confirmActionWithOption({
      title: "删除运维资料？", confirmLabel: "删除资料", tone: "danger",
      message: "“" + displayName + "”及其尚未发布的 AIOps 版本记录将被删除，此操作不可撤销。KC 审计记录会保留。",
      option: { label: "同时删除物理文件", description: "仅删除未被其他知识版本引用的 KC 源文件；共享文件会保留。", defaultSelected: false },
    });
    return { confirmed: result.confirmed, purgeSourceObjects: result.optionSelected };
  }

  async function runDetailAction(action) {
    if (!state.detail || state.submittingReview) return;
    if (action === "download") return downloadSource();
    if (action === "upload-version") {
      document.getElementById("detail-drawer").close();
      resetUpload(state.detail.asset);
      document.getElementById("upload-drawer").showModal();
      return;
    }
    if (action === "delete") {
      const confirmation = await confirmAssetDeletion(state.detail.asset.display_name);
      if (!confirmation.confirmed) return;
      const purgeQuery = confirmation.purgeSourceObjects ? "?purge_source_objects=true" : "";
      await KBotAIOpsAuth.request(`${api}/assets/${encodeURIComponent(state.detail.asset.asset_id)}${purgeQuery}`, { method: "DELETE", headers: { "If-Match": `"rv-${state.detail.asset.row_version}"` } });
      shell.toast(confirmation.purgeSourceObjects ? "资料已删除，可安全清理的源文件已物理删除" : "资料已删除，KC 源文件已保留");
      document.getElementById("detail-drawer").close();
      state.detail = null;
      await loadAssets();
      return;
    }
    const detail = state.detail;
    const version = detail.version;
    const strategy = document.getElementById("manual-parse-strategy")?.value;
    if (action === "approve" && !state.detail.parse_strategy_options?.some(option => option.selectable && option.strategy === strategy)) return;
    const options = action === "reconcile" ? { method: "POST" } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
      expected_row_version: Number(version.row_version), comment: null, parse_strategy: action === "approve" ? strategy : undefined }) };
    const submitsTagging = action === "approve" || (action === "retry" && version.status === "TAGGING_FAILED");
    const button = document.querySelector(`#detail-actions [data-review="${action}"]`);
    state.submittingReview = true;
    if (button) button.disabled = true;
    try {
      await KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(version.asset_version_id)}:${action}`, options);
      if (state.detail === detail) {
        document.getElementById("detail-drawer").close();
        state.detail = null;
      }
      shell.toast(submitsTagging ? "标签识别已提交，可在资料列表查看结果" : "知识版本状态已更新");
    } finally {
      state.submittingReview = false;
      if (button) button.disabled = action === "approve" && !document.getElementById("manual-parse-strategy")?.value;
    }
    await loadAssets();
  }

  shell.ready.then(async () => {
    document.getElementById("upload-drawer").addEventListener("cancel", event => { if (state.uploading) event.preventDefault(); });
    document.getElementById("upload-knowledge-base").onchange = () => { renderUploadLedger(); renderUploadTaggingHint(); };
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    document.getElementById("open-upload").onclick = () => { resetUpload(); document.getElementById("upload-drawer").showModal(); };
    document.getElementById("refresh-knowledge").onclick = () => loadAssets().catch((error) => shell.toast(error.message));
    document.getElementById("knowledge-search").oninput = renderAssets;
    document.getElementById("knowledge-status").onchange = () => loadAssets().catch((error) => shell.toast(error.message));
    document.getElementById("knowledge-base-filter").onchange = () => loadAssets().catch((error) => shell.toast(error.message));
    document.getElementById("knowledge-assets").onclick = (event) => { const button = event.target.closest("[data-asset-id]"); if (button) openAsset(button.dataset.assetId).catch((error) => shell.toast(error.message)); };
    document.getElementById("choose-files").onclick = () => document.getElementById("manual-file").click();
    document.getElementById("manual-file").onchange = (event) => { addFiles(Array.from(event.target.files || [])); event.target.value = ""; };
    const zone = document.getElementById("upload-drop-zone");
    ["dragenter", "dragover"].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.add("dragging"); }));
    ["dragleave", "drop"].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.remove("dragging"); }));
    zone.addEventListener("drop", (event) => addFiles(Array.from(event.dataTransfer?.files || [])));
    document.getElementById("clear-files").onclick = () => { state.entries = []; renderUploadLedger(); };
    document.getElementById("manual-name").oninput = updateSubmitUpload;
    document.getElementById("upload-files").oninput = (event) => {
      if (state.uploading || !event.target.matches("[data-file-name]")) return;
      const row = event.target.closest("[data-entry-id]");
      const entry = state.entries.find(item => item.id === row?.dataset.entryId);
      if (entry && !entry.started) entry.name = event.target.value;
      updateSubmitUpload();
    };
    document.getElementById("upload-files").onclick = (event) => { const button = event.target.closest("[data-remove-file]"); if (!button) return; const row = button.closest("[data-entry-id]"); state.entries = state.entries.filter((item) => item.id !== row.dataset.entryId); renderUploadLedger(); };
    document.getElementById("manual-form").onsubmit = (event) => submitUpload(event).catch((error) => { document.getElementById("manual-result").textContent = error.message; shell.toast(error.message); });
    document.getElementById("detail-tabs").onclick = (event) => { const button = event.target.closest("[data-tab]"); if (!button) return; state.detailTab = button.dataset.tab; renderDetailBody(); };
    document.getElementById("detail-body").onclick = (event) => { if (event.target.closest("[data-download-source]")) downloadSource(event.target.closest("[data-download-source]").dataset.downloadSource).catch((error) => shell.toast(error.message)); };
    document.getElementById("detail-actions").onclick = (event) => { const review = event.target.closest("[data-review]"); const upload = event.target.closest("[data-upload-version]"); const download = event.target.closest("[data-download-source]"); const remove = event.target.closest("[data-delete-asset]"); const action = review?.dataset.review || (upload ? "upload-version" : download ? "download" : remove ? "delete" : ""); if (action) runDetailAction(action).catch((error) => shell.toast(error.message)); };
    setInterval(() => {
      if (!document.hidden && !loadingAssets && state.assets.some(item => isProcessing(item.status))) {
        loadAssets().catch(() => { /* 手动刷新时呈现错误，后台刷新保留当前列表。 */ });
      }
    }, 5000);
    try { await Promise.all([loadTags(), loadKnowledgeBases()]); await loadAssets(); } catch (error) { shell.toast(error.message); }
  });
})();
