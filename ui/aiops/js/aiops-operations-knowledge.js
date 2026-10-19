(function () {
  "use strict";

  const api = "/api/v1/apps/aiops/operations-knowledge";
  const shell = globalThis.KBotAIOpsShell;
  const state = {
    assets: [],
    detail: null,
    detailTab: "overview",
    groupingMode: "EACH_FILE",
    entries: [],
    uploadAsset: null,
    tags: [],
    selectedTagIds: [],
    modelCatalog: [],
  };

  const escape = (value) => shell.escape(value == null ? "" : value);
  const newId = () => KBotAIOpsAuth.uuid();
  const kindLabel = (value) => value === "MANUAL" ? "运维手册" : "诊断案例";
  const tagCategories = [["RESOURCE_CATEGORY", "资源分类"], ["TECHNOLOGY_STACK", "技术栈"], ["VENDOR", "厂商"]];
  const statusLabel = (value) => ({ TAGGING: "标签识别中", TAGGING_FAILED: "标签识别失败", PROCESSING: "解析中", DRAFT: "待发布审核", REVIEW_REQUIRED: "待审批", PUBLISHED: "已发布", RETIRED: "已退役", REJECTED: "已拒绝", FAILED: "解析失败" }[value] || value || "—");
  const statusTone = (value) => ["FAILED", "TAGGING_FAILED", "REJECTED"].includes(value) ? "bad" : ["TAGGING", "PROCESSING", "DRAFT", "REVIEW_REQUIRED"].includes(value) ? "warn" : ["PUBLISHED", "READY"].includes(value) ? "good" : "";
  const badge = (value, label = statusLabel(value)) => `<span class="ops-badge ${statusTone(value)}">${escape(label)}</span>`;
  const reviewLabel = (value) => value === "TAGGING" ? "AI 识别中" : value === "TAGGING_FAILED" ? "识别失败" : value === "REVIEW_REQUIRED" ? "待审批" : ["PROCESSING", "PUBLISHED", "RETIRED"].includes(value) ? "已通过" : value === "REJECTED" ? "已拒绝" : value === "DRAFT" ? "待审核" : "—";
  const parseLabel = (value) => ["TAGGING", "TAGGING_FAILED", "REVIEW_REQUIRED"].includes(value) ? "未开始" : value === "PROCESSING" ? "解析中" : value === "FAILED" ? "解析失败" : ["PUBLISHED", "RETIRED"].includes(value) ? "解析完成" : "—";
  const formatBytes = (bytes) => bytes < 1024 ? `${bytes} B` : bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(bytes < 10240 ? 1 : 0)} KB` : `${(bytes / 1024 ** 2).toFixed(bytes < 10 * 1024 ** 2 ? 1 : 0)} MB`;
  const totalBytes = () => state.entries.reduce((sum, item) => sum + item.file.size, 0);
  const isBundle = () => !state.uploadAsset && state.groupingMode === "SINGLE_BUNDLE";

  const digest = (file) => KBotSha256.file(file);

  function fillModelSelect(id, category, selected, emptyLabel) {
    const select = document.getElementById(id);
    const options = state.modelCatalog
      .filter((item) => Number(item.category) === category)
      .map((item) => `<option value="${escape(item.model_id)}" ${String(item.model_id) === String(selected || "") ? "selected" : ""}>${escape(item.display_name)} · ${escape(item.provider)}</option>`)
      .join("");
    select.innerHTML = `<option value="">${escape(emptyLabel)}</option>${options}`;
  }

  async function openModelConfiguration() {
    const [catalog, configured] = await Promise.all([
      KBotAIOpsAuth.request("/api/v1/model-catalog"),
      KBotAIOpsAuth.request(`${api}/collection-models`),
    ]);
    state.modelCatalog = Array.isArray(catalog) ? catalog : [];
    const models = configured.models || {};
    fillModelSelect("model-parser-ocr", 6, models.parser_ocr, "请选择 OCR 模型");
    fillModelSelect("model-parser-vlm", 5, models.parser_vlm, "不启用视觉解析");
    fillModelSelect("model-parser-llm", 1, models.parser_llm, "请选择 LLM 解析模型");
    fillModelSelect("model-retrieval-llm", 1, models.retrieval_llm, "请选择检索规划模型");
    fillModelSelect("model-embedding", 2, models.embedding, "请选择文本向量模型");
    fillModelSelect("model-visual-embedding", 3, models.visual_embedding, "不启用视觉向量");
    document.getElementById("model-config-result").textContent = "";
    document.getElementById("model-config-drawer").showModal();
  }

  async function saveModelConfiguration(event) {
    event.preventDefault();
    const roles = {
      parser_ocr: "model-parser-ocr",
      parser_vlm: "model-parser-vlm",
      parser_llm: "model-parser-llm",
      retrieval_llm: "model-retrieval-llm",
      embedding: "model-embedding",
      visual_embedding: "model-visual-embedding",
    };
    const models = Object.fromEntries(Object.entries(roles)
      .map(([role, id]) => [role, document.getElementById(id).value])
      .filter(([, value]) => Boolean(value)));
    const button = document.getElementById("save-model-config");
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(`${api}/collection-models`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ models }),
      });
      document.getElementById("model-config-drawer").close();
      shell.toast("运维知识库解析模型已保存");
    } finally {
      button.disabled = false;
    }
  }

  function assetRow(item) {
    const action = item.status === "REVIEW_REQUIRED" ? "审批标签" : ["TAGGING", "PROCESSING"].includes(item.status) ? "查看进度" : ["FAILED", "TAGGING_FAILED"].includes(item.status) ? "处理异常" : item.status === "DRAFT" ? "审核" : "查看";
    return `<tr><td><button class="knowledge-primary-link" type="button" data-asset-id="${escape(item.asset_id)}">${escape(item.display_name)}</button><small>${escape(kindLabel(item.asset_kind))} · L${escape(item.security_level)} · ${escape(item.asset_id)}</small></td><td>${badge(item.status, reviewLabel(item.status))}</td><td>${badge(item.status, parseLabel(item.status))}</td><td><time>${escape(shell.fmt(item.created_at))}</time>${item.updated_at !== item.created_at ? `<small>更新于 ${escape(shell.fmt(item.updated_at))}</small>` : ""}</td><td><button type="button" data-asset-id="${escape(item.asset_id)}">${action}</button></td></tr>`;
  }

  function renderAssets() {
    const keyword = document.getElementById("knowledge-search").value.trim().toLowerCase();
    const items = keyword ? state.assets.filter((item) => [item.display_name, item.asset_id, kindLabel(item.asset_kind)].some((value) => String(value).toLowerCase().includes(keyword))) : state.assets;
    document.getElementById("knowledge-count").textContent = `${items.length} 项`;
    document.getElementById("knowledge-assets").innerHTML = items.length ? items.map(assetRow).join("") : '<tr><td colspan="5" class="ops-empty">当前条件下没有运维资料。</td></tr>';
  }

  async function loadAssets() {
    const query = new URLSearchParams({ limit: "200" });
    const status = document.getElementById("knowledge-status").value;
    if (status) query.set("status", status);
    const result = await KBotAIOpsAuth.request(`${api}/assets?${query}`);
    state.assets = result.items || [];
    renderAssets();
  }

  async function loadTags() {
    const result = await KBotAIOpsAuth.request(`${api}/tags?include_disabled=true`);
    state.tags = result.items || [];
  }

  function stageClass(stage, status) {
    if (stage === "uploaded") return "is-complete";
    if (stage === "tagged") return status === "TAGGING" ? "is-current" : status === "TAGGING_FAILED" ? "is-error" : "is-complete";
    if (stage === "approved") return ["DRAFT", "REVIEW_REQUIRED"].includes(status) ? "is-current" : status === "REJECTED" ? "is-error" : ["PROCESSING", "FAILED", "PUBLISHED", "RETIRED"].includes(status) ? "is-complete" : "";
    if (stage === "parsed") return status === "PROCESSING" ? "is-current" : status === "FAILED" ? "is-error" : ["PUBLISHED", "RETIRED"].includes(status) ? "is-complete" : "";
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
    const editable = detail.version.status === "REVIEW_REQUIRED";
    const catalog = tagCatalog(detail);
    const groups = tagCategories.map(([category, label]) => {
      const tags = catalog.filter((item) => item.category === category && (item.status === "ACTIVE" || state.selectedTagIds.includes(item.tag_id) || suggestions.has(item.tag_id)));
      const options = tags.length ? tags.map((tag) => {
        const suggestion = suggestions.get(tag.tag_id);
        const evidence = (suggestion?.evidence || []).map((item) => `<article class="knowledge-tag-evidence"><small>${escape(item.file_name || "原始文件")}</small><q>${escape(item.quote)}</q></article>`).join("");
        return `<label class="knowledge-tag-option${suggestion ? " suggested" : ""}"><input type="checkbox" data-tag-id="${escape(tag.tag_id)}" ${state.selectedTagIds.includes(tag.tag_id) ? "checked" : ""} ${editable ? "" : "disabled"}><span><strong>${escape(tag.display_name)}</strong><small>${escape(tag.description || "暂无说明")}</small>${suggestion ? `<em>AI 建议 · ${Math.round(Number(suggestion.confidence || 0) * 100)}%</em>` : ""}</span></label>${evidence}`;
      }).join("") : '<p class="ops-empty">该分类尚未配置有效标签。</p>';
      return `<fieldset class="knowledge-tag-group"><legend>${escape(label)}</legend>${options}</fieldset>`;
    }).join("");
    const failure = analysis.status === "FAILED" ? `<div class="knowledge-warning knowledge-error"><strong>标签识别失败</strong><p>${escape(analysis.error_summary || analysis.error_code || "未知错误")}</p></div>` : "";
    const warnings = analysis.warnings?.length ? `<div class="knowledge-warning"><strong>识别提示</strong><p>${escape(analysis.warnings.join("；"))}</p></div>` : "";
    return `<section class="knowledge-form-section knowledge-tag-review"><header><b>02</b><div><strong>受控标签</strong><small>${editable ? "核对 AI 原文证据并调整选择；批准后标签随版本冻结。" : "展示当前版本已识别或审批保存的标签。"}</small></div></header>${failure}${groups}${warnings}</section>`;
  }

  function renderOverview(detail) {
    const status = detail.version.status;
    const scope = detail.scopes.map((item) => `${item.kind}=${item.value}`).join("；") || "通用";
    return `<section class="knowledge-stage-track"><div class="${stageClass("uploaded", status)}"><span>1</span><strong>上传</strong></div><div class="${stageClass("tagged", status)}"><span>2</span><strong>标签识别</strong></div><div class="${stageClass("approved", status)}"><span>3</span><strong>审批</strong></div><div class="${stageClass("parsed", status)}"><span>4</span><strong>KC 解析</strong></div><div class="${stageClass("published", status)}"><span>5</span><strong>发布</strong></div></section><section class="knowledge-form-section"><header><b>01</b><div><strong>资料概览</strong><small>当前资料及其最新知识版本。</small></div></header><dl class="knowledge-summary-grid"><div><dt>资料类型</dt><dd>${escape(kindLabel(detail.asset.asset_kind))}</dd></div><div><dt>安全级</dt><dd>L${escape(detail.asset.security_level)}</dd></div><div><dt>当前版本</dt><dd>v${escape(detail.version.version_no)}</dd></div><div><dt>受控标签</dt><dd>${escape(scope)}</dd></div><div><dt>提交时间</dt><dd>${escape(shell.fmt(detail.asset.created_at))}</dd></div><div><dt>最近更新</dt><dd>${escape(shell.fmt(detail.asset.updated_at))}</dd></div></dl></section>${renderTags(detail)}${detail.version.extraction_warnings?.length ? `<section class="knowledge-warning"><strong>解析提示</strong><ul>${detail.version.extraction_warnings.map((item) => `<li>${escape(item)}</li>`).join("")}</ul></section>` : ""}`;
  }

  function renderFiles(detail) {
    const sources = detail.sources?.length ? detail.sources.map((source, index) => `<article class="knowledge-source-card"><span>${Number(source.locator?.ordinal ?? index) + 1}</span><div><strong>${escape(source.external_id || detail.asset.display_name)}</strong><small>${source.locator?.role === "SUPPLEMENT" ? "补充材料" : "正文内容"}${source.locator?.required_flag ? " · 必要成员" : ""}</small><code>${escape(source.content_hash)}</code></div></article>`).join("") : '<p class="ops-empty">尚未登记原始文件。</p>';
    const indexes = detail.indexes?.length ? detail.indexes.map((item) => `<tr><td><code>${escape(item.index_ref_id)}</code></td><td>${badge(item.status, item.status)}</td><td>${escape(item.error_summary || "—")}</td></tr>`).join("") : '<tr><td colspan="3" class="ops-empty">尚未登记索引结果。</td></tr>';
    return `<section class="knowledge-form-section"><header><b>01</b><div><strong>原始文件</strong><small>展示当前版本包含的文件及其 Bundle 角色。</small></div></header><div class="knowledge-source-list">${sources}</div><div class="knowledge-source-actions"><button type="button" data-download-source>下载正文文件</button></div><dl class="knowledge-technical-list"><dt>Bundle 摘要</dt><dd><code>${escape(detail.version.source_hash)}</code></dd></dl></section><section class="knowledge-form-section"><header><b>02</b><div><strong>索引状态</strong><small>用于判断当前版本是否已经形成可检索知识。</small></div></header><div class="ops-table-wrap"><table class="ops-table"><thead><tr><th>索引</th><th>状态</th><th>异常</th></tr></thead><tbody>${indexes}</tbody></table></div></section>${detail.version.profile ? `<section class="knowledge-form-section"><header><b>03</b><div><strong>提炼结果</strong><small>结构化结果仅辅助导航，运维结论仍应以原文为准。</small></div></header><pre class="knowledge-profile">${escape(JSON.stringify(detail.version.profile, null, 2))}</pre></section>` : ""}`;
  }

  function renderHistory(detail) {
    const rows = detail.reviews?.length ? detail.reviews.map((item) => `<li><div><strong>${escape(item.decision)}</strong><small>${escape(item.reviewer_id)}</small></div><time>${escape(shell.fmt(item.created_at))}</time></li>`).join("") : '<p class="ops-empty">当前版本还没有审核记录。</p>';
    return `<section class="knowledge-form-section"><header><b>01</b><div><strong>处理记录</strong><small>保留当前版本的审核决策与责任人。</small></div></header><ol class="knowledge-timeline">${rows}</ol></section>`;
  }

  function renderDetailBody() {
    if (!state.detail) return;
    document.getElementById("detail-body").innerHTML = state.detailTab === "overview" ? renderOverview(state.detail) : state.detailTab === "files" ? renderFiles(state.detail) : renderHistory(state.detail);
    document.querySelectorAll("#detail-tabs [data-tab]").forEach((button) => button.classList.toggle("active", button.dataset.tab === state.detailTab));
  }

  function renderDetailActions(detail) {
    const actions = ['<button type="button" data-download-source>下载正文文件</button>'];
    if (detail.asset.asset_kind === "MANUAL") actions.push('<button type="button" data-upload-version>上传新版本</button>');
    if (["TAGGING", "PROCESSING"].includes(detail.version.status)) actions.push('<button type="button" data-review="reconcile">立即处理</button>');
    if (["DRAFT", "REVIEW_REQUIRED", "TAGGING_FAILED"].includes(detail.version.status)) actions.push('<button class="danger" type="button" data-review="reject">拒绝</button>');
    if (detail.asset.asset_kind === "MANUAL" && detail.version.status === "REVIEW_REQUIRED") actions.push(`<button class="primary" type="button" data-review="approve" ${state.selectedTagIds.length ? "" : "disabled"}>批准并开始解析</button>`);
    if (detail.asset.asset_kind === "DIAGNOSIS_CASE" && ["DRAFT", "REVIEW_REQUIRED"].includes(detail.version.status)) actions.push('<button class="primary" type="button" data-review="publish">审核通过并发布</button>');
    if (detail.version.status === "PUBLISHED") actions.push('<button class="danger" type="button" data-review="retire">退役</button>');
    if (["FAILED", "TAGGING_FAILED"].includes(detail.version.status)) actions.push(`<button class="primary" type="button" data-review="retry">${detail.version.status === "TAGGING_FAILED" ? "重试标签识别" : "重试解析"}</button>`);
    document.getElementById("detail-actions").innerHTML = actions.join("");
  }

  async function openAsset(assetId) {
    const asset = await KBotAIOpsAuth.request(`${api}/assets/${encodeURIComponent(assetId)}`);
    const version = asset.versions?.[0];
    if (!version) throw new Error("该资料尚未形成知识版本");
    state.detail = await KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(version.asset_version_id)}`);
    const approved = state.detail.version.tag_analysis?.approved_tag_ids || state.detail.scopes.map((item) => item.tag_id).filter(Boolean);
    state.selectedTagIds = approved.length ? approved : (state.detail.version.tag_analysis?.suggestions || []).map((item) => item.tag_id);
    state.detailTab = "overview";
    document.getElementById("detail-eyebrow").textContent = `${kindLabel(state.detail.asset.asset_kind)} · v${state.detail.version.version_no}`;
    document.getElementById("detail-title").textContent = state.detail.asset.display_name;
    document.getElementById("detail-subtitle").textContent = state.detail.asset.asset_id;
    document.getElementById("detail-status").innerHTML = badge(state.detail.version.status);
    renderDetailBody();
    renderDetailActions(state.detail);
    document.getElementById("detail-drawer").showModal();
  }

  function renderUploadLedger() {
    const ledger = document.getElementById("upload-ledger");
    ledger.hidden = !state.entries.length;
    document.getElementById("upload-total").textContent = `${state.entries.length} 个 · ${formatBytes(totalBytes())}`;
    document.getElementById("upload-files").innerHTML = state.entries.map((entry, index) => `<article class="knowledge-file-row" data-entry-id="${escape(entry.id)}"><span>${index + 1}</span><div><strong>${escape(entry.file.name)}</strong><small>${escape(entry.file.type || "未知类型")} · ${formatBytes(entry.file.size)}</small></div>${isBundle() ? `<div class="knowledge-file-settings"><label><span>文件角色</span><select data-role><option value="CONTENT" ${entry.role === "CONTENT" ? "selected" : ""}>正文内容</option><option value="SUPPLEMENT" ${entry.role === "SUPPLEMENT" ? "selected" : ""}>补充材料</option></select></label><label class="knowledge-required"><input data-required type="checkbox" ${entry.required ? "checked" : ""}><span>必要成员</span></label></div>` : '<span class="ops-badge">单个文件</span>'}<button type="button" data-remove-file>移除</button></article>`).join("");
    document.getElementById("submit-upload").disabled = !state.entries.length || totalBytes() > 64 * 1024 * 1024 || state.entries.length > 20;
  }

  function setGroupingMode(mode) {
    state.groupingMode = mode;
    if (mode === "EACH_FILE" && state.entries.length > 1) state.entries = state.entries.slice(0, 1);
    document.querySelectorAll(".knowledge-mode-picker label").forEach((label) => label.classList.toggle("selected", label.querySelector("input").value === mode));
    document.getElementById("manual-file").multiple = mode === "SINGLE_BUNDLE";
    document.getElementById("manual-name-label").textContent = mode === "SINGLE_BUNDLE" ? "Bundle 名称" : "资料名称";
    document.getElementById("manual-name").placeholder = mode === "SINGLE_BUNDLE" ? "例如：Oracle RAC 故障处置手册包" : "例如：Oracle Data Guard 运维手册";
    document.getElementById("file-step-help").textContent = mode === "SINGLE_BUNDLE" ? "成员顺序、角色和必要性会一同写入 Bundle。" : "原文会保留，可在详情中下载核验。";
    document.getElementById("drop-help").textContent = mode === "SINGLE_BUNDLE" ? "可多次选择并追加文件，最多 20 个、总计 64 MB。" : "每次上传一个文件，重新选择会替换当前文件。";
    renderUploadLedger();
  }

  function addFiles(files) {
    const selected = state.groupingMode === "EACH_FILE" ? files.slice(0, 1) : files;
    if (state.groupingMode === "EACH_FILE") {
      state.entries = selected.map((file) => ({ id: newId(), file, role: "CONTENT", required: false }));
      if (selected[0] && !document.getElementById("manual-name").value.trim()) document.getElementById("manual-name").value = selected[0].name.replace(/\.[^.]+$/, "") || selected[0].name;
    } else {
      const known = new Set(state.entries.map(({ file }) => `${file.name}:${file.size}:${file.lastModified}`));
      state.entries.push(...selected.filter((file) => !known.has(`${file.name}:${file.size}:${file.lastModified}`)).map((file) => ({ id: newId(), file, role: "CONTENT", required: false })));
    }
    document.getElementById("manual-result").textContent = state.entries.length > 20 || totalBytes() > 64 * 1024 * 1024 ? "请将文件数量控制在 20 个以内，总大小不超过 64 MB。" : "";
    renderUploadLedger();
  }

  function resetUpload(asset = null) {
    state.uploadAsset = asset;
    state.entries = [];
    document.getElementById("manual-form").reset();
    document.getElementById("manual-result").textContent = "";
    document.getElementById("manual-name").value = asset?.display_name || "";
    document.getElementById("manual-security").value = String(asset?.security_level || 1);
    document.getElementById("upload-mode-section").hidden = Boolean(asset);
    document.getElementById("upload-title").textContent = asset ? "上传新版本" : "上传资料";
    document.getElementById("upload-subtitle").textContent = asset ? "新版本完成解析和审核前，当前发布版本仍继续参与检索。" : "可上传单个文件，或将多个相互关联的文件组成同一个 Bundle。";
    document.getElementById("file-step").textContent = asset ? "01" : "02";
    document.getElementById("metadata-step").textContent = asset ? "02" : "03";
    setGroupingMode("EACH_FILE");
  }

  async function submitUpload(event) {
    event.preventDefault();
    if (!state.entries.length) return;
    const name = document.getElementById("manual-name").value.trim();
    if (!name) return;
    const button = document.getElementById("submit-upload");
    const metadata = { display_name: name, security_level: Number(document.getElementById("manual-security").value) };
    button.disabled = true;
    document.getElementById("manual-result").textContent = "正在登记原文并识别受控标签…";
    try {
      if (state.uploadAsset) {
        const file = state.entries[0].file;
        const hash = await digest(file);
        await KBotAIOpsAuth.request(`${api}/assets/${encodeURIComponent(state.uploadAsset.asset_id)}/versions`, { method: "POST", headers: { "Content-Type": file.type || "application/octet-stream", "Idempotency-Key": newId(), "X-File-Name": encodeURIComponent(file.name), "X-Content-SHA256": hash, "X-Upload-Metadata": encodeURIComponent(JSON.stringify(metadata)), "X-Expected-Asset-Row-Version": String(state.uploadAsset.row_version) }, body: file });
      } else {
        const declarations = await Promise.all(state.entries.map(async (entry, ordinal) => ({ part_name: `file-${ordinal}`, client_file_id: entry.id, display_name: entry.file.name, declared_mime_type: entry.file.type || "application/octet-stream", byte_size: entry.file.size, content_sha256: await digest(entry.file), ordinal, role: isBundle() ? entry.role : "CONTENT", required_flag: isBundle() ? entry.required : false })));
        const body = new FormData();
        body.set("grouping_mode", state.groupingMode);
        body.set("metadata", JSON.stringify(metadata));
        body.set("files", JSON.stringify(declarations));
        if (isBundle()) body.set("bundle", JSON.stringify({ client_bundle_id: newId(), title: name, security_level: metadata.security_level }));
        declarations.forEach((item, index) => body.append(item.part_name, state.entries[index].file));
        const receipt = await KBotAIOpsAuth.request(`${api}/uploads`, { method: "POST", headers: { "Idempotency-Key": newId() }, body });
        const rejected = (receipt.items || []).filter((item) => item.status === "REJECTED");
        if (rejected.length) throw new Error(rejected.map((item) => item.message).filter(Boolean).join("；") || "部分资料未被受理");
      }
      document.getElementById("upload-drawer").close();
      shell.toast(isBundle() ? "Bundle 已受理，正在识别受控标签" : "资料已受理，正在识别受控标签");
      await loadAssets();
    } finally {
      button.disabled = false;
    }
  }

  async function downloadSource() {
    if (!state.detail) return;
    await KBotAIOpsAuth.download(`${api}/versions/${encodeURIComponent(state.detail.version.asset_version_id)}/source`, state.detail.asset.display_name, "*/*");
  }

  async function runDetailAction(action) {
    if (!state.detail) return;
    if (action === "download") return downloadSource();
    if (action === "upload-version") {
      document.getElementById("detail-drawer").close();
      resetUpload(state.detail.asset);
      document.getElementById("upload-drawer").showModal();
      return;
    }
    const version = state.detail.version;
    const options = action === "reconcile" ? { method: "POST" } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ expected_row_version: Number(version.row_version), comment: null, tag_ids: action === "approve" ? state.selectedTagIds : [] }) };
    state.detail = await KBotAIOpsAuth.request(`${api}/versions/${encodeURIComponent(version.asset_version_id)}:${action}`, options);
    shell.toast("知识版本状态已更新");
    document.getElementById("detail-drawer").close();
    await loadAssets();
  }

  shell.ready.then(async () => {
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    document.getElementById("open-upload").onclick = () => { resetUpload(); document.getElementById("upload-drawer").showModal(); };
    document.getElementById("open-model-config").onclick = () => openModelConfiguration().catch((error) => shell.toast(error.message));
    document.getElementById("model-config-form").onsubmit = (event) => saveModelConfiguration(event).catch((error) => { document.getElementById("model-config-result").textContent = error.message; shell.toast(error.message); });
    document.getElementById("refresh-knowledge").onclick = () => loadAssets().catch((error) => shell.toast(error.message));
    document.getElementById("knowledge-search").oninput = renderAssets;
    document.getElementById("knowledge-status").onchange = () => loadAssets().catch((error) => shell.toast(error.message));
    document.getElementById("knowledge-assets").onclick = (event) => { const button = event.target.closest("[data-asset-id]"); if (button) openAsset(button.dataset.assetId).catch((error) => shell.toast(error.message)); };
    document.querySelectorAll('[name="grouping-mode"]').forEach((input) => { input.onchange = () => setGroupingMode(input.value); });
    document.getElementById("choose-files").onclick = () => document.getElementById("manual-file").click();
    document.getElementById("manual-file").onchange = (event) => { addFiles(Array.from(event.target.files || [])); event.target.value = ""; };
    const zone = document.getElementById("upload-drop-zone");
    ["dragenter", "dragover"].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.add("dragging"); }));
    ["dragleave", "drop"].forEach((name) => zone.addEventListener(name, (event) => { event.preventDefault(); zone.classList.remove("dragging"); }));
    zone.addEventListener("drop", (event) => addFiles(Array.from(event.dataTransfer?.files || [])));
    document.getElementById("clear-files").onclick = () => { state.entries = []; renderUploadLedger(); };
    document.getElementById("upload-files").onchange = (event) => { const row = event.target.closest("[data-entry-id]"); const entry = state.entries.find((item) => item.id === row?.dataset.entryId); if (!entry) return; if (event.target.matches("[data-role]")) entry.role = event.target.value; if (event.target.matches("[data-required]")) entry.required = event.target.checked; };
    document.getElementById("upload-files").onclick = (event) => { const button = event.target.closest("[data-remove-file]"); if (!button) return; const row = button.closest("[data-entry-id]"); state.entries = state.entries.filter((item) => item.id !== row.dataset.entryId); renderUploadLedger(); };
    document.getElementById("manual-form").onsubmit = (event) => submitUpload(event).catch((error) => { document.getElementById("manual-result").textContent = error.message; shell.toast(error.message); });
    document.getElementById("detail-tabs").onclick = (event) => { const button = event.target.closest("[data-tab]"); if (!button) return; state.detailTab = button.dataset.tab; renderDetailBody(); };
    document.getElementById("detail-body").onclick = (event) => { if (event.target.closest("[data-download-source]")) runDetailAction("download").catch((error) => shell.toast(error.message)); };
    document.getElementById("detail-body").onchange = (event) => { const input = event.target.closest("[data-tag-id]"); if (!input) return; state.selectedTagIds = input.checked ? [...new Set([...state.selectedTagIds, input.dataset.tagId])] : state.selectedTagIds.filter((value) => value !== input.dataset.tagId); renderDetailActions(state.detail); };
    document.getElementById("detail-actions").onclick = (event) => { const review = event.target.closest("[data-review]"); const upload = event.target.closest("[data-upload-version]"); const download = event.target.closest("[data-download-source]"); const action = review?.dataset.review || (upload ? "upload-version" : download ? "download" : ""); if (action) runDetailAction(action).catch((error) => shell.toast(error.message)); };
    try { await Promise.all([loadTags(), loadAssets()]); } catch (error) { shell.toast(error.message); }
  });
})();
