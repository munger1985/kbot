(function () {
  "use strict";

  const api = "/api/v1/apps/aiops/operations-knowledge";
  const shell = globalThis.KBotAIOpsShell;
  const state = {
    assets: [],
    detail: null,
    detailTab: "overview",
    sourcePreview: null,
    entries: [],
    uploadAsset: null,
    knowledgeBases: [],
    uploading: false,
    submittingReview: false,
    batchIds: new Set(),
    batch: null,
  };

  const isProcessing = (status) => ["TAGGING", "PROCESSING"].includes(status);
  let assetRequest = 0;
  let loadingAssets = false;
  const escape = (value) => shell.escape(value == null ? "" : value);
  const newId = () => KBotAIOpsAuth.uuid();
  const kindLabel = (value) => value === "MANUAL" ? "运维手册" : "诊断案例";
  const tagCategories = [["RESOURCE_CATEGORY", "资源分类"], ["TECHNOLOGY_STACK", "技术栈"], ["VENDOR", "厂商"]];
  const strategyLabels = { TEXT: "原生解析", AUTO: "自动判断", VISUAL: "视觉增强" };
  const statusLabel = (value) => ({ TAGGING: "标签识别中", TAGGING_FAILED: "标签识别失败", PROCESSING: "解析中", DRAFT: "待发布审核", REVIEW_REQUIRED: "待审批", PUBLISHED: "已发布", RETIRED: "已退役", REJECTED: "已拒绝", FAILED: "解析失败" }[value] || value || "—");
  const statusTone = (value) => ["FAILED", "TAGGING_FAILED", "REJECTED"].includes(value) ? "bad" : ["TAGGING", "PROCESSING", "DRAFT", "REVIEW_REQUIRED"].includes(value) ? "warn" : ["PUBLISHED", "READY"].includes(value) ? "good" : "";
  const badge = (value, label = statusLabel(value)) => `<span class="status-pill ${({ good: "status-success", bad: "status-danger", warn: "status-warning" })[statusTone(value)] || "status-muted"}">${escape(label)}</span>`;
  const reviewLabel = (value) => value === "REVIEW_REQUIRED" ? "待审批" : ["TAGGING", "TAGGING_FAILED", "PROCESSING", "PUBLISHED", "RETIRED"].includes(value) ? "已通过" : value === "REJECTED" ? "已拒绝" : value === "DRAFT" ? "待审核" : "—";
  const parseLabel = (value, kind = "MANUAL") => value === "RETIRED" ? statusLabel(value) : value === "TAGGING" ? "AI 识别中" : value === "TAGGING_FAILED" ? "AI 识别失败" : value === "REVIEW_REQUIRED" || (kind === "MANUAL" && ["DRAFT", "REJECTED"].includes(value)) ? "未开始" : value === "PROCESSING" ? "解析中" : value === "FAILED" ? "解析失败" : value === "PUBLISHED" ? "解析完成" : "—";
  const formatBytes = (bytes) => bytes < 1024 ? `${bytes} B` : bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(bytes < 10240 ? 1 : 0)} KB` : `${(bytes / 1024 ** 2).toFixed(bytes < 10 * 1024 ** 2 ? 1 : 0)} MB`;
  const pendingUploads = () => state.entries.filter(entry => !entry.accepted);
  const totalBytes = () => pendingUploads().reduce((sum, item) => sum + item.file.size, 0);
  const uploadLocked = () => state.entries.some(entry => entry.started);
  const selectedKnowledgeBaseId = () => document.getElementById("knowledge-base-filter").value;

  const digest = (file) => KBotSha256.file(file);

  function assetRow(item) {
    const action = item.status === "REVIEW_REQUIRED" ? "审批资料" : ["TAGGING", "PROCESSING"].includes(item.status) ? "查看进度" : ["FAILED", "TAGGING_FAILED"].includes(item.status) ? "处理异常" : item.status === "DRAFT" ? "审核" : "查看";
    const selection = item.status === "REVIEW_REQUIRED" ? `<input type="checkbox" data-batch-id="${escape(item.asset_id)}" aria-label="选择待审批资料：${escape(item.display_name)}"${state.batchIds.has(item.asset_id) ? " checked" : ""}>` : "";
    return `<tr><td class="knowledge-select-cell">${selection}</td><td><button class="table-primary-button" type="button" data-asset-id="${escape(item.asset_id)}">${escape(item.display_name)}</button><small class="table-secondary">${escape(kindLabel(item.asset_kind))} · L${escape(item.security_level)} · ${escape(item.asset_id)}</small></td><td>${escape(state.knowledgeBases.find(base => base.knowledge_base_id === item.knowledge_base_id)?.display_name || "知识库已不可用")}</td><td>${badge(item.status, reviewLabel(item.status))}</td><td>${badge(item.status, parseLabel(item.status, item.asset_kind))}</td><td><span class="knowledge-submitted-time">${escape(shell.fmt(item.created_at))}</span>${item.updated_at !== item.created_at ? `<small class="table-secondary">更新于 ${escape(shell.fmt(item.updated_at))}</small>` : ""}</td><td><button class="row-action" type="button" data-asset-id="${escape(item.asset_id)}">${action}</button></td></tr>`;
  }

  function visibleAssets() {
    const keyword = document.getElementById("knowledge-search").value.trim().toLowerCase();
    return keyword ? state.assets.filter((item) => [item.display_name, item.asset_id, kindLabel(item.asset_kind)].some((value) => String(value).toLowerCase().includes(keyword))) : state.assets;
  }

  const pendingAssets = () => visibleAssets().filter(item => item.status === "REVIEW_REQUIRED");

  function syncBatchSelection() {
    const pending = pendingAssets();
    const selected = pending.filter(item => state.batchIds.has(item.asset_id)).length;
    const all = document.getElementById("knowledge-select-all");
    all.disabled = loadingAssets || !pending.length;
    all.checked = pending.length > 0 && selected === pending.length;
    all.indeterminate = selected > 0 && selected < pending.length;
    document.getElementById("open-batch-review").disabled = loadingAssets || selected === 0;
    document.getElementById("knowledge-batch-count").textContent = selected ? `已选 ${selected} 项` : "";
  }

  function renderAssets() {
    const items = visibleAssets();
    const eligible = new Set(items.filter(item => item.status === "REVIEW_REQUIRED").map(item => item.asset_id));
    state.batchIds = new Set([...state.batchIds].filter(id => eligible.has(id)));
    document.getElementById("knowledge-count").textContent = `${items.length} 项`;
    document.getElementById("knowledge-assets").innerHTML = items.length ? items.map(assetRow).join("") : '<tr><td colspan="7" class="empty-cell">当前条件下没有运维资料。</td></tr>';
    syncBatchSelection();
  }

  async function loadAssets() {
    const knowledgeBaseId = selectedKnowledgeBaseId();
    const query = new URLSearchParams({ limit: "200", asset_kind: "MANUAL" });
    if (knowledgeBaseId) query.set("knowledge_base_id", knowledgeBaseId);
    const status = document.getElementById("knowledge-status").value;
    if (status) query.set("status", status);
    const requestId = ++assetRequest;
    loadingAssets = true;
    syncBatchSelection();
    try {
      const result = await KBotAIOpsAuth.request(`${api}/assets?${query}`);
      if (requestId !== assetRequest) return;
      state.assets = result.items || [];
      renderAssets();
    } finally {
      if (requestId === assetRequest) {
        loadingAssets = false;
        syncBatchSelection();
      }
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

  function renderTags(detail) {
    if (detail.asset.asset_kind !== "MANUAL") return "";
    const analysis = detail.version.tag_analysis || {};
    const suggestions = new Map((analysis.suggestions || []).map((item) => [item.tag_id, item]));
    const assigned = new Set(analysis.assigned_tag_ids || []);
    const recognized = new Map();
    for (const scope of detail.scopes || []) {
      if (!scope.tag_id) continue;
      assigned.add(scope.tag_id);
      recognized.set(scope.tag_id, { tag_id: scope.tag_id, category: scope.kind, display_name: scope.value });
    }
    for (const tag of analysis.vocabulary || []) {
      if (assigned.has(tag.tag_id) || suggestions.has(tag.tag_id)) recognized.set(tag.tag_id, tag);
    }
    for (const tag of suggestions.values()) {
      recognized.set(tag.tag_id, { ...recognized.get(tag.tag_id), ...tag });
    }
    const groups = tagCategories.map(([category, label]) => {
      const tags = [...recognized.values()].filter((item) => item.category === category);
      if (!tags.length) return "";
      const options = tags.map((tag) => {
        const suggestion = suggestions.get(tag.tag_id);
        return `<label class="knowledge-tag-option${suggestion ? " suggested" : ""}"><input type="checkbox" checked disabled><span><strong>${escape(tag.display_name)}</strong><small>${escape(tag.description || "暂无说明")}</small>${suggestion ? `<em>AI 识别 · ${Math.round(Number(suggestion.confidence || 0) * 100)}%</em>` : ""}</span></label>`;
      }).join("");
      return `<fieldset class="knowledge-tag-group"><legend>${escape(label)}</legend>${options}</fieldset>`;
    }).join("");
    const failure = analysis.status === "FAILED" ? `<div class="knowledge-warning knowledge-error"><strong>标签识别失败</strong><p>${escape(analysis.error_summary || analysis.error_code || "未知错误")}</p></div>` : "";
    const warnings = analysis.warnings?.length ? `<div class="knowledge-warning"><strong>识别提示</strong><p>${escape(analysis.warnings.join("；"))}</p></div>` : "";
    const hint = ({ REVIEW_REQUIRED: "资料审批通过后才会调用 AI 识别标签。", REJECTED: "该版本已被拒绝，不会调用 AI 识别标签。", TAGGING: "AI 正在根据原文证据识别受控标签。", TAGGING_FAILED: "AI 标签识别未完成，可在修复原因后重试。" })[detail.version.status] || "展示 AI 根据原文证据自动冻结的标签。";
    const empty = !groups && !failure ? '<p class="ops-empty">暂无识别标签。</p>' : "";
    return `<section class="drawer-form-section knowledge-tag-review"><header><span>02</span><div><strong>受控标签</strong><small>${hint}</small></div></header>${failure}${groups}${empty}${warnings}</section>`;
  }

  function renderOverview(detail) {
    const status = detail.version.status;
    const caseScope = detail.asset.asset_kind === "DIAGNOSIS_CASE" ? detail.scopes.map((item) => `${item.kind}=${item.value}`).join("；") || "通用" : "";
    const stages = detail.asset.asset_kind === "MANUAL"
      ? `<section class="knowledge-stage-track"><div class="${stageClass("uploaded", status)}"><span>1</span><strong>上传</strong></div><div class="${stageClass("approved", status)}"><span>2</span><strong>审批</strong></div><div class="${stageClass("tagged", status)}"><span>3</span><strong>标签识别</strong></div><div class="${stageClass("parsed", status)}"><span>4</span><strong>文档解析</strong></div><div class="${stageClass("published", status)}"><span>5</span><strong>发布</strong></div></section>`
      : `<section class="knowledge-stage-track four-steps"><div class="${stageClass("uploaded", status, "DIAGNOSIS_CASE")}"><span>1</span><strong>生成</strong></div><div class="${stageClass("parsed", status, "DIAGNOSIS_CASE")}"><span>2</span><strong>文档解析</strong></div><div class="${stageClass("approved", status, "DIAGNOSIS_CASE")}"><span>3</span><strong>审批</strong></div><div class="${stageClass("published", status, "DIAGNOSIS_CASE")}"><span>4</span><strong>发布</strong></div></section>`;
    return `${stages}<section class="drawer-form-section"><header><span>01</span><div><strong>资料概览</strong><small>当前资料及其最新知识版本。</small></div></header><dl class="knowledge-summary-grid"><div><dt>资料类型</dt><dd>${escape(kindLabel(detail.asset.asset_kind))}</dd></div><div><dt>安全级</dt><dd>L${escape(detail.asset.security_level)}</dd></div><div><dt>当前版本</dt><dd>v${escape(detail.version.version_no)}</dd></div>${caseScope ? `<div><dt>适用范围</dt><dd>${escape(caseScope)}</dd></div>` : ""}<div><dt>提交时间</dt><dd>${escape(shell.fmt(detail.asset.created_at))}</dd></div><div><dt>最近更新</dt><dd>${escape(shell.fmt(detail.asset.updated_at))}</dd></div></dl></section>${renderTags(detail)}${detail.version.extraction_warnings?.length ? `<section class="knowledge-warning"><strong>解析提示</strong><ul>${detail.version.extraction_warnings.map((item) => `<li>${escape(item)}</li>`).join("")}</ul></section>` : ""}`;
  }

  const manualSources = (detail) => (detail.sources || []).filter(source => source.source_kind === "MANUAL_FILE");
  const sourceDocumentId = (source, detail) => {
    const id = source.locator?.document_version_id;
    return id && detail.indexes?.some(index => index.document_version_id === id) ? id : "";
  };
  const indexLabel = (status) => ({ PENDING_APPROVAL: "待审批", PROCESSING: "处理中", READY: "索引已就绪", FAILED: "索引失败" })[status] || status;

  function renderSourcePreview(id, fileName) {
    const preview = state.sourcePreview;
    if (!id || preview?.id !== id) return "";
    const content = preview.kind === "text" ? `<pre>${escape(preview.content)}</pre>${preview.truncated ? "<small>仅预览前 64 KiB，下载可查看完整内容。</small>" : ""}`
      : preview.kind === "image" ? `<img src="${escape(preview.url)}" alt="${escape(fileName)}">`
      : preview.kind === "pdf" ? `<iframe src="${escape(preview.url)}#toolbar=0" title="当前版本 PDF 原文预览" referrerpolicy="no-referrer"></iframe>`
      : `<p role="status">${({ loading: "正在读取原文件…", unavailable: "此文件不支持在线预览，请下载查看。", error: "无法读取原文件，请重试预览或下载。" })[preview.kind]}</p>`;
    return `<div class="knowledge-source-preview" aria-label="原文预览"><button type="button" class="row-action" data-close-source-preview>收起预览</button>${content}</div>`;
  }

  function renderManualProfile(profile) {
    if (profile?.schema_version !== "OPS_MANUAL_PROFILE.v1") return "";
    const scope = profile.scope;
    const fields = [
      ["发布方", profile.source?.publisher], ["原文版本", profile.source?.document_version],
      ["主题", (profile.topics?.length ? profile.topics : scope?.topic)?.join("、")],
      ["资源分类", scope?.resource_category?.join("、")], ["技术栈", scope?.technology_stack?.join("、")], ["厂商", scope?.vendor?.join("、")],
    ].filter(([, value]) => value).map(([label, value]) => `<div><dt>${label}</dt><dd>${escape(value)}</dd></div>`).join("");
    const procedures = (profile.procedures || []).map(procedure => `<details><summary>${escape(procedure.title)}</summary><ol>${procedure.steps.map(step => `<li><span>${escape(step.description)}</span>${step.command_text ? `<pre>${escape(step.command_text)}</pre>` : ""}</li>`).join("")}</ol></details>`).join("");
    const warnings = (profile.warnings || []).map(warning => `<li>${escape(warning)}</li>`).join("");
    return `<section class="drawer-form-section"><header><span>03</span><div><strong>提炼结果</strong><small>用于快速浏览，运维结论仍以原文为准。</small></div></header>${fields ? `<dl class="knowledge-profile-fields">${fields}</dl>` : ""}${procedures ? `<div class="knowledge-procedures">${procedures}</div>` : ""}${warnings ? `<ul class="knowledge-warning">${warnings}</ul>` : ""}${!fields && !procedures && !warnings ? '<p class="ops-empty">未提炼出可展示的主题或操作步骤，请查看原文。</p>' : ""}</section>`;
  }

  function renderFiles(detail) {
    const files = manualSources(detail);
    const sources = files.map(source => {
      const id = sourceDocumentId(source, detail);
      const name = source.external_id || detail.asset.display_name;
      const actions = id ? `<div class="knowledge-source-actions"><button class="secondary-button" type="button" data-preview-source="${escape(id)}">预览原文</button><button class="secondary-button" type="button" data-download-source="${escape(id)}">下载原文</button></div>` : "<small>原文件未关联到当前文档版本，无法预览或下载。</small>";
      return `<article class="knowledge-source-card"><strong>${escape(name)}</strong>${actions}${renderSourcePreview(id, name)}</article>`;
    }).join("") || '<p class="ops-empty">尚未登记上传原文。</p>';
    const indexes = (detail.indexes || []).map(item => {
      const name = files.find(source => source.locator?.document_version_id === item.document_version_id)?.external_id || "文档索引";
      const error = item.error_summary || item.error_code;
      return `<li><div><strong>${escape(name)}</strong>${badge(item.status, indexLabel(item.status))}</div>${item.last_checked_at ? `<small>检查于 ${escape(shell.fmt(item.last_checked_at))}</small>` : ""}${error ? `<p>${escape(error)}</p>` : ""}</li>`;
    }).join("");
    const status = indexes ? `<ul class="knowledge-index-list">${indexes}</ul>` : ["PUBLISHED", "RETIRED"].includes(detail.version.status)
      ? '<p class="knowledge-warning">该版本有发布记录，但没有关联当前索引，不能参与检索。可上传原文件作为新版本，重新审批和发布。</p>'
      : '<p class="ops-empty">当前版本还没有索引记录。</p>';
    const metadata = detail.version.source_hash ? `<details class="knowledge-source-metadata"><summary>技术信息</summary><p>资料内容指纹：<code>${escape(detail.version.source_hash)}</code></p></details>` : "";
    return `<section class="drawer-form-section"><header><span>01</span><div><strong>原始文件</strong><small>预览或下载当前版本实际上传的文件。</small></div></header><div class="knowledge-source-list">${sources}</div>${metadata}</section><section class="drawer-form-section"><header><span>02</span><div><strong>索引状态</strong><small>显示当前版本的索引记录，和历史发布状态分开。</small></div></header>${status}</section>${renderManualProfile(detail.version.profile)}`;
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
      const options = (detail.parse_strategy_options || []).map(option => {
        const reason = option.executor_unavailable ? "模型解析执行端未启用" : (option.unavailable_roles || []).map(role => role.toUpperCase()).join("、") + " 不可用";
        return `<option value="${escape(option.strategy)}"${option.selectable ? "" : " disabled"}>${escape(strategyLabels[option.strategy] + (option.selectable ? "" : `（${reason}）`))}</option>`;
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

  async function fetchAssetDetail(assetId) {
    const asset = await KBotAIOpsAuth.request(`${api}/assets/${encodeURIComponent(assetId)}`);
    const version = asset.versions?.[0];
    if (!version) throw new Error("该资料尚未形成知识版本");
    return KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(version.asset_version_id)}`);
  }

  async function openAsset(assetId) {
    clearSourcePreview();
    state.detail = await fetchAssetDetail(assetId);
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
    document.getElementById("upload-files").innerHTML = state.entries.map((entry, index) => {
      const name = escape(entry.file.name);
      const status = entry.accepted ? "已接收，等待审批" : entry.error ? "提交失败" : "待提交";
      const statusClass = entry.accepted ? "status-success" : entry.error ? "status-danger" : "status-muted";
      return `<article class="upload-file-row${state.uploadAsset ? " upload-file-row--version" : ""}" data-entry-id="${escape(entry.id)}"><span class="upload-file-order">${index + 1}</span><div class="upload-file-identity"><strong title="${name}">${name}</strong><div class="upload-file-meta"><small>${escape(entry.file.type || "未知类型")} · ${formatBytes(entry.file.size)}</small><span class="status-pill ${statusClass}">${status}</span></div></div>
      ${state.uploadAsset ? "" : `<label class="control-field upload-file-name"><span class="field-label">资料名称</span><input data-file-name value="${escape(entry.name)}" maxlength="256"${entry.started ? " disabled" : disabled}></label>`}
      <button class="row-action danger-row-action upload-file-remove" type="button" aria-label="移除 ${name}" data-remove-file${disabled}>移除</button>${entry.error ? `<p class="upload-file-error">${escape(entry.error)}</p>` : ""}</article>`;
    }).join("");
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

  function sourcePath(detail, id) {
    return `${api}/versions/${encodeURIComponent(detail.version.asset_version_id)}/source?${new URLSearchParams({ document_version_id: id })}`;
  }

  function clearSourcePreview() {
    state.sourcePreview?.controller.abort();
    if (state.sourcePreview?.url) URL.revokeObjectURL(state.sourcePreview.url);
    state.sourcePreview = null;
  }

  async function previewSource(id) {
    const detail = state.detail;
    if (!detail || !manualSources(detail).some(source => sourceDocumentId(source, detail) === id)) return;
    clearSourcePreview();
    const preview = { id, controller: new AbortController(), kind: "loading" };
    state.sourcePreview = preview;
    renderDetailBody();
    try {
      const blob = await KBotAIOpsAuth.requestBlob(sourcePath(detail, id), { signal: preview.controller.signal, headers: { Accept: "*/*" } });
      if (state.sourcePreview !== preview) return;
      const mime = blob.type.split(";")[0].trim().toLowerCase();
      if (/^text\/(?:plain|markdown|csv)$/.test(mime)) {
        const limit = 64 * 1024;
        preview.content = await blob.slice(0, limit).text();
        preview.truncated = blob.size > limit;
        preview.kind = "text";
      } else {
        const pdf = mime === "application/pdf" && navigator.pdfViewerEnabled !== false && await blob.slice(0, 5).text() === "%PDF-";
        if (state.sourcePreview !== preview) return;
        if (pdf || /^image\/(?:png|jpeg|webp)$/.test(mime)) {
          preview.kind = pdf ? "pdf" : "image";
          preview.url = URL.createObjectURL(blob.slice(0, blob.size, mime));
        } else preview.kind = "unavailable";
      }
    } catch { preview.kind = "error"; }
    if (state.sourcePreview === preview) renderDetailBody();
  }

  async function downloadSource(id) {
    const detail = state.detail;
    if (!detail) return;
    const source = manualSources(detail).find(item => sourceDocumentId(item, detail) === id);
    if (!source) return;
    await KBotAIOpsAuth.download(sourcePath(detail, id), source.external_id || detail.asset.display_name, "*/*");
  }

  async function confirmAssetDeletion(displayName) {
    const result = await shell.confirmActionWithOption({
      title: "删除运维资料？", confirmLabel: "删除资料", tone: "danger",
      message: "“" + displayName + "”及其尚未发布的 AIOps 版本记录将被删除，此操作不可撤销。KC 审计记录会保留。",
      option: { label: "同时删除物理文件", description: "仅删除未被其他知识版本引用的 KC 源文件；共享文件会保留。", defaultSelected: false },
    });
    return { confirmed: result.confirmed, purgeSourceObjects: result.optionSelected };
  }

  async function mapBatch(items, run) {
    const results = [];
    for (let offset = 0; offset < items.length; offset += 4) {
      results.push(...await Promise.all(items.slice(offset, offset + 4).map(async item => {
        try { return { item, value: await run(item) }; }
        catch (error) { return { item, error: error?.message || "未确认成功，请刷新后核对" }; }
      })));
    }
    return results;
  }

  function renderBatchReview(batch) {
    const preparing = batch.phase === "preparing";
    const submitting = batch.phase === "submitting";
    const ready = batch.phase === "ready";
    const status = document.getElementById("batch-review-status");
    status.textContent = preparing ? "正在核对待审批版本和可用解析模式…" : submitting ? "正在提交审批…" : batch.notice;
    document.getElementById("batch-review-title").textContent = `批量审批 · ${batch.assets.length} 项`;
    document.getElementById("batch-review-items").innerHTML = batch.assets.map(asset => `<li>${escape(asset.display_name)}</li>`).join("");
    const failures = document.getElementById("batch-review-failures");
    document.getElementById("batch-review-failure-section").hidden = !batch.failures.length;
    failures.innerHTML = batch.failures.map(({item, error}) => `<li>${escape(item.display_name)}：${escape(error)}</li>`).join("");
    document.getElementById("batch-review-strategy-field").hidden = !ready && !submitting;
    const picker = document.getElementById("batch-review-strategy");
    picker.disabled = !ready;
    picker.innerHTML = '<option value="" disabled>选择解析模式</option>' +
      batch.strategies.map(value => `<option value="${value}">${strategyLabels[value]}</option>`).join("");
    picker.value = batch.strategy;
    const confirm = document.getElementById("submit-batch-review");
    confirm.hidden = !ready && !submitting;
    confirm.disabled = !ready || !batch.strategies.includes(batch.strategy);
    document.querySelectorAll("#batch-review-drawer [data-batch-close]").forEach(button => { button.disabled = submitting; });
  }

  async function openBatchReview() {
    const assets = pendingAssets().filter(item => state.batchIds.has(item.asset_id));
    if (loadingAssets || !assets.length) return;
    const batch = { assets, phase: "preparing", entries: [], strategies: [], strategy: "", notice: "", failures: [] };
    state.batch = batch;
    renderBatchReview(batch);
    const drawer = document.getElementById("batch-review-drawer");
    drawer.showModal();
    drawer.querySelector(".admin-drawer-body").scrollTop = 0;
    const results = await mapBatch(assets, async asset => {
      const detail = await fetchAssetDetail(asset.asset_id);
      if (detail.version?.status !== "REVIEW_REQUIRED") throw new Error("资料已不处于待审批状态，请刷新后重试");
      return detail;
    });
    if (state.batch !== batch) return;
    batch.failures = results.filter(result => result.error);
    batch.entries = results.filter(result => result.value);
    batch.strategies = Object.keys(strategyLabels).filter(strategy => batch.entries.every(({value}) =>
      value.parse_strategy_options?.some(option => option.strategy === strategy && option.selectable)));
    batch.phase = batch.failures.length || !batch.strategies.length ? "blocked" : "ready";
    batch.strategy = batch.strategies.includes("AUTO") ? "AUTO" : "";
    batch.notice = batch.failures.length ? "有资料无法审批，请关闭后刷新或取消勾选这些资料。" :
      !batch.strategies.length ? "所选资料没有共同可用的解析模式，请分开审批。" : "";
    renderBatchReview(batch);
  }

  async function submitBatchReview() {
    const batch = state.batch;
    if (batch?.phase !== "ready" || !batch.strategies.includes(batch.strategy)) return;
    batch.phase = "submitting";
    renderBatchReview(batch);
    const results = await mapBatch(batch.entries.map(entry => ({ ...entry.item, version: entry.value.version })), async item => {
      const version = item.version;
      return KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(version.asset_version_id)}:approve`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
          expected_row_version: Number(version.row_version), comment: null, parse_strategy: batch.strategy,
        }),
      });
    });
    batch.failures = results.filter(result => result.error);
    const accepted = results.length - batch.failures.length;
    results.filter(result => !result.error).forEach(({item}) => state.batchIds.delete(item.asset_id));
    batch.phase = "done";
    batch.notice = `已受理 ${accepted} 项，未确认受理 ${batch.failures.length} 项。`;
    if (batch.failures.length) renderBatchReview(batch);
    else {
      state.batch = null;
      document.getElementById("batch-review-drawer").close();
      shell.toast(`已提交 ${accepted} 项审批，标签识别将在后台执行`);
    }
    await loadAssets();
  }

  async function runDetailAction(action) {
    if (!state.detail || state.submittingReview) return;
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
      const result = await KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(version.asset_version_id)}:${action}`, options);
      if (action === "retire") {
        const { asset } = result;
        const status = document.getElementById("knowledge-status").value;
        state.assets = state.assets.flatMap(item => item.asset_id !== asset.asset_id ? [item]
          : status && status !== asset.status ? [] : [asset]);
        renderAssets();
      }
      if (state.detail === detail) {
        document.getElementById("detail-drawer").close();
        state.detail = null;
      }
      shell.toast(action === "retire" ? "资料已退役" : submitsTagging ? "标签识别已提交，可在资料列表查看结果" : "知识版本状态已更新");
    } finally {
      state.submittingReview = false;
      if (button) button.disabled = action === "approve" && !document.getElementById("manual-parse-strategy")?.value;
    }
    await loadAssets();
  }

  shell.ready.then(async () => {
    const batchDrawer = document.getElementById("batch-review-drawer");
    batchDrawer.addEventListener("cancel", event => {
      if (state.batch?.phase === "submitting") event.preventDefault();
    });
    batchDrawer.addEventListener("close", () => { if (!batchDrawer.open) state.batch = null; });
    document.querySelectorAll("[data-batch-close]").forEach(button => { button.onclick = () => batchDrawer.close(); });
    document.getElementById("knowledge-select-all").onchange = (event) => {
      state.batchIds = new Set(event.target.checked ? pendingAssets().map(item => item.asset_id) : []);
      document.querySelectorAll("[data-batch-id]").forEach(input => { input.checked = state.batchIds.has(input.dataset.batchId); });
      syncBatchSelection();
    };
    document.getElementById("knowledge-assets").onchange = (event) => {
      const id = event.target.dataset.batchId;
      if (!id) return;
      if (event.target.checked) state.batchIds.add(id);
      else state.batchIds.delete(id);
      syncBatchSelection();
    };
    document.getElementById("open-batch-review").onclick = () => openBatchReview().catch((error) => shell.toast(error.message));
    document.getElementById("batch-review-strategy").onchange = (event) => {
      if (state.batch?.phase !== "ready") return;
      state.batch.strategy = event.target.value;
      document.getElementById("submit-batch-review").disabled = !state.batch.strategies.includes(state.batch.strategy);
    };
    document.getElementById("submit-batch-review").onclick = () => submitBatchReview().catch((error) => shell.toast(error.message));
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
    document.getElementById("detail-drawer").addEventListener("close", clearSourcePreview);
    document.getElementById("detail-tabs").onclick = (event) => { const button = event.target.closest("[data-tab]"); if (!button || state.detailTab === button.dataset.tab) return; clearSourcePreview(); state.detailTab = button.dataset.tab; renderDetailBody(); };
    document.getElementById("detail-body").onclick = (event) => {
      const preview = event.target.closest("[data-preview-source]");
      const download = event.target.closest("[data-download-source]");
      if (preview) void previewSource(preview.dataset.previewSource);
      else if (download) downloadSource(download.dataset.downloadSource).catch((error) => shell.toast(error.message));
      else if (event.target.closest("[data-close-source-preview]")) { clearSourcePreview(); renderDetailBody(); }
    };
    document.getElementById("detail-actions").onclick = (event) => { const review = event.target.closest("[data-review]"); const upload = event.target.closest("[data-upload-version]"); const remove = event.target.closest("[data-delete-asset]"); const action = review?.dataset.review || (upload ? "upload-version" : remove ? "delete" : ""); if (action) runDetailAction(action).catch((error) => shell.toast(error.message)); };
    setInterval(() => {
      if (!document.hidden && !loadingAssets && state.assets.some(item => isProcessing(item.status))) {
        loadAssets().catch(() => { /* 手动刷新时呈现错误，后台刷新保留当前列表。 */ });
      }
    }, 5000);
    try { await loadKnowledgeBases(); await loadAssets(); } catch (error) { shell.toast(error.message); }
  });
})();
