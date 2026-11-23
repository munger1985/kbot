/* 独立页面共享请求、流程控件和作用域清理；只挂载当前页面的操作。 */
(function () {
  "use strict";
  const api = KBotKnowledgeApi, shell = KBotKnowledgeShell, esc = shell.escapeHtml;
  const view = document.body.dataset.page;
  const pdfReaderUrl = new URL("../vendor/pdf-search-reader.js", location.href).href;
  const directoryView = ["processing", "files"].includes(view);
  const management = view === "knowledge-cores";
  const linkQuery = new URLSearchParams(location.search);
  const labels = {
    ACTIVE: "已启用", DISABLED: "已停用", RECEIVED: "等待解析", PARSING: "解析中", PARSED: "解析完成",
    READY: "索引就绪", RETRY: "索引待再次处理", FAILED: "失败", INDEX_FAILED: "索引失败", CANCELLED: "已取消", PENDING: "待审核",
    AWAITING_APPROVAL: "等待审批",
    APPROVED: "已通过", REJECTED: "已拒绝", PUBLISHED: "已发布", WITHDRAWN: "已撤回",
    COMPLETE: "已完成", PARTIAL: "部分覆盖", EMPTY: "无结果",
    NO_MATCH: "没有命中", MODALITY_UNAVAILABLE: "本次查询没有可用通道", CHANNEL_FAILED: "检索通道执行失败",
    IMAGE_VERIFICATION_LIMIT: "仅核验排名靠前的候选",
    TEXT: "文字", FIGURE: "图示", PHOTO: "图片", TABLE: "表格", PAGE: "页面", REGION: "区域",
    READ: "读取", UPLOAD: "上传", REVIEW: "审核", MANAGE: "管理", PUBLISH: "发布", WITHDRAW: "撤回",
  };
  const label = value => labels[value] || value || "—";
  const taskLabel = value => value === "FAILED" ? "解析失败" : label(value);
  const parseResult = row => row.parse_status === "PARSED" ? "成功" : row.parse_status === "FAILED" ? "失败" : "未完成";
  const taskResult = row => row.parse_status === "PARSED" && row.processing_status === "INDEX_FAILED"
    ? (row.status === "PUBLISHED" ? "解析成功 · 更新失败" : "解析成功 · 索引失败 · 未发布") : parseResult(row);
  const taskProgress = {
    AWAITING_APPROVAL: ["等待审批"], PARSE_QUEUED: ["排队等候解析"], PREPARING: ["准备解析"],
    EXTRACTING: ["识别内容"], ORGANIZING: ["整理内容"],
    PARSE_RETRY: ["解析待重试", "retry"], INDEX_QUEUED: ["等待建立索引"],
    INDEXING: ["正在建索引"], INDEX_RETRY: ["索引待重试", "retry"],
    INDEX_CONFIRMING: ["校验索引"], READY: ["索引已就绪", "ready"],
    PARSE_FAILED: ["解析失败", "failed"], INDEX_FAILED: ["建索引失败", "failed"],
    CANCELLED: ["已取消"], REJECTED: ["审批未通过"],
    REBUILD_QUEUED: ["等待重建索引"], REBUILDING: ["正在重建索引"], REBUILD_RETRY: ["重建待重试", "retry"],
    REBUILD_CONFIRMING: ["校验重建索引"], REBUILD_READY: ["重建索引已校验", "ready"],
    REBUILD_COMPLETE: ["重建完成 · 已生效", "ready"], REBUILD_FAILED: ["重建索引失败", "failed"],
  };
  const activeProgress = new Set(["AWAITING_APPROVAL", "PARSE_QUEUED", "PREPARING", "EXTRACTING", "ORGANIZING",
    "PARSE_RETRY", "INDEX_QUEUED", "INDEXING", "INDEX_RETRY", "INDEX_CONFIRMING",
    "REBUILD_QUEUED", "REBUILDING", "REBUILD_RETRY", "REBUILD_CONFIRMING", "REBUILD_READY"]);
  function renderProgress({ progress_stage: stage, index_progress: index }, readyLabel) {
    if (!taskProgress[stage]) return "—";
    const [stageLabel, tone = ""] = taskProgress[stage];
    const label = stage === "READY" && readyLabel ? readyLabel : stageLabel;
    const total = index?.total_batches, done = index?.committed_batches ?? 0;
    const known = Number.isSafeInteger(total) && total > 0 && done <= total;
    const pct = known && ["READY", "REBUILD_READY", "REBUILD_COMPLETE"].includes(stage) && index.validated && done === total ? 100 :
      known && done < total ? Math.min(99, Math.floor(done * 100 / total)) : null;
    const count = known ? `${done} / ${total} 批已提交` : `已提交 ${done} 批`;
    const notes = index ? [count, !known && stage === "INDEXING" ? "正在计算总批次" : ""].filter(Boolean).join(" · ") :
      { EXTRACTING: "正在提取正文与页面内容", ORGANIZING: "正在整理内容结构", PARSE_FAILED: "未开始建索引" }[stage];
    const track = !index ? "" : `<span class="knowledge-task-track" role="progressbar" aria-label="索引批次完成度"
      aria-valuetext="${esc([label, notes, pct === null ? "" : pct + "%"].filter(Boolean).join("，"))}"
      ${pct === null ? "" : `aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}" style="--task-progress:${pct}%"`}
      data-track="${pct !== null ? "numeric" : known ? "confirming" : "unknown"}"></span>`;
    return `<div class="knowledge-task-progress"${tone ? ` data-tone="${tone}"` : ""}${index ? " data-index-progress" : ""}>
      <strong>${label}</strong>${notes ? `<span class="knowledge-task-count">${esc(notes)}</span>` : ""}${track}
      ${pct !== null ? `<b class="knowledge-task-percent">${pct}%</b>` : ""}</div>`;
  }
  function renderBatchDetail(index, summary = "批次明细") {
    if (!index) return "";
    const done = Number(index.committed_batches), total = Number(index.total_batches);
    const submitted = !done ? "尚无已提交批次" : done === 1 ? "第 1 批已提交" : `第 1–${done} 批已提交`;
    const remaining = total > done ? `<li>第 ${total === done + 1 ? total : `${done + 1}–${total}`} 批尚未提交</li>` : "";
    return `<details class="knowledge-task-batches"><summary>${summary}</summary><ul><li>${submitted}</li>${remaining}
      ${total ? "" : "<li>总批次尚未确认</li>"}</ul></details>`;
  }
  function renderRebuildProgress(item) {
    const index = item.index_progress;
    if (!index) return "";
    const stage = item.status === "CANCELLED" ? "CANCELLED" : item.status === "FAILED" ? "INDEX_FAILED" : item.status === "READY" ? "READY" :
      item.status === "PENDING" ? (item.error_code ? "INDEX_RETRY" : "INDEX_QUEUED") :
      index.total_batches && index.committed_batches === index.total_batches ? "INDEX_CONFIRMING" : "INDEXING";
    return renderProgress({ progress_stage: stage, index_progress: index }, "重建索引已校验");
  }
  const indexStage = row => row.processing_status === "INDEX_FAILED" ? "失败" :
    row.generation_ready ? "已就绪" : row.processing_status === "RETRY" ? "待再次处理" : "未就绪";
  const maintenanceHint = row => row.parse_status === "FAILED" ? "内容没有提取成功，可先重新解析。" :
    row.processing_status === "INDEX_FAILED" ? "内容已经提取成功，只需重试建索引。" :
    row.generation_ready ? "重新解析会重新读取原文件并建索引，适用于解析规则或 OCR、VLM 模型有变化时。重建索引复用已有解析结果，不重新读取原文件。" : "";
  const indexBatchLabel = batch => batch?.status === "PUBLISHED" && batch.source_published === false
    ? "已自动发布" : rebuildLabels[batch?.status];
  const strategyLabels = { TEXT: "原生解析", AUTO: "自动判断", VISUAL: "视觉增强" };
  const canApprove = row => row.parse_status !== "AWAITING_APPROVAL" ||
    row.parse_strategy_options?.some(option => option.selectable && option.strategy === state.parseStrategy);
  const canReprocess = row => state.reprocessMode === "ORIGINAL" ? row.original_parse_available :
    row.parse_strategy_options?.some(option => option.selectable && option.strategy === state.parseStrategy);
  const strategyPicker = row => `<select name="parseStrategy" aria-label="解析方案">${emptyStrategyOption(state.parseStrategy)}${(row.parse_strategy_options || []).map(option => {
    const reason = option.executor_unavailable ? "模型解析执行端未启用" : (option.unavailable_roles || []).map(role => role.toUpperCase()).join("、") + " 不可用";
    return `<option value="${esc(option.strategy)}"${option.selectable ? "" : " disabled"}${state.parseStrategy === option.strategy ? " selected" : ""}>${esc(strategyLabels[option.strategy] + (option.selectable ? "" : `（${reason}）`))}</option>`;
  }).join("")}</select>`;
  const parseRoles = [["ocr", 6, "OCR"], ["vlm", 5, "VLM"]];
  const indexRoles = [["embedding", 2, "文本 Embedding"], ["visual_embedding", 3, "视觉 Embedding"]];
  const indexNames = { embedding: "文本 Embedding", visual_embedding: "视觉 Embedding" };
  const rebuildLabels = { PENDING: "等待处理", RUNNING: "处理中", READY: "索引就绪", FAILED: "失败", CANCELLED: "已取消",
    STALE: "资料或配置已变化", PUBLISHED: "已切换", DISCARDED: "已放弃" };
  const activeIndexBatch = batch => batch && !["PUBLISHED", "DISCARDED"].includes(batch.status);
  const changedIndexes = () => indexRoles.filter(([role]) => (state[role] || "") !== (state.managed?.index_models?.[role] || ""));
  const canIndex = (...ops) => appCan("knowledge_core_manage") && can(library(), ...ops);
  const modelsChanged = (roles, group) => roles.some(([role]) => (state[role] || null) !== (state.managed?.[group]?.[role] || null));
  const publication = value => ["PUBLISHED", "WITHDRAWN"].includes(value) ? label(value) : "未发布";
  const pendingFileReview = row => row.review === "PENDING" &&
    !["PUBLISHED", "WITHDRAWN", "CANCELLED"].includes(row.status) && row.processing_status !== "CANCELLED";
  const versionTime = value => value ? new Date(value).toLocaleString("zh-CN") : "—";
  const locationOf = value => (value?.passage_display?.pages || []).map(page => `第 ${page} 页`).join(" · ")
    || [...new Set((value?.locators || []).map(part => part.page_no ? `第 ${part.page_no} 页` : "").filter(Boolean))].join(" · ")
    || (value?.passage_key ? "文内段落" : "暂无页码");
  const enc = encodeURIComponent;
  const deleteReasons = { HAS_DOCUMENTS: "该库仍有资料，不能删除。", AGENT_REFERENCE: "该库仍被当前或历史 Agent 版本引用，不能删除。" };
  const matrix = globalThis.KBotKnowledgeMemberMatrix;
  const state = { libraries: [], libraryId: "", epoch: 0, busy: false, searchTreeRevision: 0, urls: new Set(), controller: new AbortController() };
  const slowSearchSeconds = 12;
  let searchWait = null;
  let directoryAction = 0, progressTimer, progressLoading = false, appliedProgressQuery = "";
  let host, dialog, access, trigger;
  let pdfViewport, pdfReader, pdfModule;
  const appCan = name => access?.permissions?.includes("knowledge_retrieval:" + name);
  const can = (library, ...operations) => operations.some(op => library?.permissions.includes(op));
  const canUpload = item => item?.status === "ACTIVE" && appCan("upload") && can(item, "UPLOAD");
  const eligible = () => state.libraries.filter(library => {
    if (management) return can(library, "MANAGE");
    if (library.status !== "ACTIVE") return false;
    if (view === "processing") return (appCan("upload") && can(library, "UPLOAD")) || (appCan("review") && can(library, "REVIEW"))
      || (appCan("knowledge_core_manage") && can(library, "MANAGE"));
    return can(library, "READ");
  });
  const library = () => state.libraries.find(value => value.library_id === state.libraryId);
  const libraryPath = () => "/libraries/" + enc(state.libraryId);
  const reviewStates = (item = library()) => {
    const review = appCan("review") && can(item, "REVIEW");
    const withdraw = appCan("knowledge_core_manage") && can(item, "WITHDRAW");
    return [...(review ? ["PENDING", "REJECTED"] : []), ...(review || withdraw ? ["PUBLISHED"] : [])];
  };
  const queueLabel = value => value === "PENDING" ? "待审批" : value ? label(value) : "全部文件";
  const memberReady = () => state.memberMatrix?.options && !state.memberMatrix.blocked && !matrix.memberWarning(state.memberMatrix);
  const membersDirty = () => Boolean(state.memberMatrix && matrix.changes(state.memberMatrix).length);
  const options = (values, selected) => values.map(([value, name]) =>
    `<option value="${esc(value)}"${value === selected ? " selected" : ""}>${esc(name)}</option>`).join("");
  const emptyStrategyOption = strategy => `<option value="" hidden disabled${strategy ? "" : " selected"}></option>`;
  function button(action, name, key = "", tone = "secondary-button", disabled = false) {
    return `<button type="button" class="${tone}" data-action="${action}" data-key="${esc(key)}"${disabled ? " disabled" : ""}>${esc(name)}</button>`;
  }
  const input = (name, value, attrs = "") => `<input name="${name}" value="${esc(value)}" ${attrs}>`;
  const select = (name, values, value) => `<select name="${name}">${options(values, value)}</select>`;
  const field = (name, body) => `<label class="control-field"><span>${esc(name)}</span>${body}</label>`;
  const libraryPicker = name => field(name, `<select data-library>${options([["", "选择知识库"], ...eligible().map(value => [value.library_id, value.display_name])], state.libraryId)}</select>`);
  const submit = (name, disabled = false, secondary = false, form = "") =>
    `<button type="submit" class="${secondary ? "secondary" : "primary"}-button"${disabled ? " disabled" : ""}${form ? ` form="${form}"` : ""}>${esc(name)}</button>`;
  const facts = (entries, card = false) => `<dl class="${card ? "knowledge-file-facts" : "knowledge-facts"}">${entries.map(([key, value]) => {
    const pair = `<dt>${esc(key)}</dt><dd>${esc(value ?? "—")}</dd>`;
    return card ? `<div>${pair}</div>` : pair;
  }).join("")}</dl>`;
  const errorHtml = () => state.error ? `<p class="table-error" role="alert">${esc(state.error)}</p>` : "";
  function target(page, version, job) {
    const query = new URLSearchParams({ library_id: state.libraryId, version_id: version });
    if (job) query.set("job_id", job);
    return `./${page}.html?${query}`;
  }
  const href = (path, name) => `<a class="secondary-button" href="${esc(path)}">${esc(name)}</a>`;

  function assertCurrent(epoch) { if (epoch !== state.epoch) throw new DOMException("工作范围已切换", "AbortError"); }
  async function request(path, options = {}) {
    const epoch = state.epoch;
    const value = await api.request("/knowledge" + path, { ...options, signal: options.signal ?? state.controller.signal });
    assertCurrent(epoch); return value;
  }
  const json = (path, method, data) => request(path, { method,
    headers: { "Content-Type": "application/json" }, body: data === undefined ? undefined : JSON.stringify(data) });
  function revoke(url) { if (url) { URL.revokeObjectURL(url); state.urls.delete(url); } }
  function objectUrl(blob) { const url = URL.createObjectURL(blob); state.urls.add(url); return url; }
  function clearPreview() { revoke(state.preview); state.preview = ""; state.previewKey = ""; }
  function clearPassage() {
    revoke(state.passagePdf);
    Object.assign(state, { passage: null, passagePdf: "", passageBefore: 2, passageAfter: 2,
      readerTab: "context", readerExpanded: false, readerPage: 1, readerZoom: 0 });
  }
  function clearFilePreview() {
    state.filePreviewController?.abort(); state.filePreviewController = null;
    if (state.filePreview?.kind !== "text") revoke(state.filePreview?.content);
    state.filePreview = null; state.filePreviewState = "";
  }
  function reset() {
    state.epoch++;
    endSearchWait();
    state.searchTreeRevision++;
    appliedProgressQuery = "";
    clearFilePreview(); clearPassage();
    state.controller.abort(); state.controller = new AbortController();
    state.urls.forEach(url => URL.revokeObjectURL(url)); state.urls.clear();
    Object.assign(state, { busy: false, catalogLoading: false, error: "", notice: "", page: 1, rows: [], more: false, mine: false,
      filterTitle: "", filterState: "", queue: "", row: null, governing: false, fileVersions: [], edit: "", managed: null, name: "", description: "", comment: "",
      batchIds: new Set(), batchStrategy: "", batchFailures: [],
      confirmingCreate: false, indexImpact: null, indexTasks: null, detailIndex: null, progressHelp: false,
      memberMatrix: null,
      preview: "", previewKey: "", result: null, anchor: "", summary: "", submittedMode: "", submittedImageIntent: null, searchOptionsOpen: false, submittedSearchOptions: "",
      historyOpen: false, historyItems: [], historyMore: false, historyId: "", historyEntry: null, historyStatus: "",
      mode: "RELEVANCE", order: "title", inputKind: "TEXT", query: "", prefix: "", descending: false, output: "TEXT", count: 10,
      createdFrom: "", createdBefore: "", queryImage: null, queryImageUrl: "", imageIntent: "SOURCE", uploads: [], uploadOpen: false, uploadError: "" });
  }
  function perform(work) {
    if (state.busy) return;
    const epoch = state.epoch;
    directoryAction++;
    state.busy = true; state.error = ""; state.notice = "";
    updateBusy();
    return Promise.resolve().then(() => { assertCurrent(epoch); return work(epoch); }).catch(error => {
      if (epoch !== state.epoch || error?.name === "AbortError") return;
      if (view === "search" && ["SCOPE_CHANGED","SNAPSHOT_STALE","CURSOR_EXPIRED","ARTIFACT_EXPIRED"].includes(error.code)) {
        state.result = null; state.anchor = ""; clearPreview(); clearPassage(); setHistoryLink(state.historyId);
        if (state.historyEntry) state.historyStatus = "这条历史的结果已变化或不可访问。请重新打开历史核对。";
        else state.error = "检索回执已失效，请重新检索。";
      } else state.error = ({
        KNOWLEDGE_LIBRARY_REVISION_CONFLICT: "知识库已变化，请刷新后重试。",
        KNOWLEDGE_TASK_REVISION_CONFLICT: "当前解析任务已变化，请关闭详情并刷新列表。",
        KNOWLEDGE_INDEX_BATCH_ACTIVE: "有互斥的索引任务尚未结束，请等待完成或先处理失败任务。",
        KNOWLEDGE_FILE_INDEX_BATCH_ACTIVE: "此文件已有未结束的重建任务，请刷新查看进度。",
        KNOWLEDGE_INDEX_BATCH_CONFLICT: "索引批次尚未就绪或已变化，请刷新索引任务。",
        KNOWLEDGE_LIBRARY_HAS_DOCUMENTS: deleteReasons.HAS_DOCUMENTS,
        KNOWLEDGE_LIBRARY_AGENT_REFERENCE: deleteReasons.AGENT_REFERENCE,
        KNOWLEDGE_LIBRARY_NAME_CONFLICT: "当前工作域已有同名知识库。",
        KNOWLEDGE_PREVIEW_UNAVAILABLE: "此版本不支持内联预览，可下载原文件。",
      })[error.code] || error?.message || "操作未完成";
    }).finally(() => {
      if (epoch === state.epoch) { state.busy = false; render(); }
    });
  }
  function updateBusy() {
    [host, dialog].forEach(root => root?.querySelectorAll("button,input,select,textarea").forEach(control => { control.disabled = true; }));
  }

  const evidence = key => state.result?.items.find(value => value.key === key);
  const isInlineText = item => item?.mime_type?.startsWith("text/") && item.byte_count <= 65536;
  const isInlineImage = item => /^image\/(png|jpeg|webp)$/.test(item?.mime_type || "");
  const canPreview = item => item?.content_ref && (isInlineText(item) || isInlineImage(item));
  const previewImage = item => state.preview && state.previewKey === item?.key
    ? `<img class="knowledge-preview" src="${esc(state.preview)}" alt="本次读取的图片证据">` : "";
  const group = key => state.result?.linked.find(value => value.anchor_key === key);
  function selectedGroups(active) {
    if (!active || !state.result) return [];
    const keys = new Set([active.key]);
    if (active.passage_key) for (const hit of state.result.primary_hits) {
      const item = evidence(hit.key);
      if (item?.version_id === active.version_id && item.passage_key === active.passage_key) keys.add(item.key);
    }
    return state.result.linked.filter(value => keys.has(value.anchor_key));
  }
  function searchDocuments() {
    const sources = new Map(), chapters = new Map(), seen = new Set();
    for (const hit of state.result?.primary_hits || []) {
      const item = evidence(hit.key);
      if (!item) continue;
      const passage = item.version_id + ":" + (item.passage_key || item.key);
      if (seen.has(passage)) continue;
      seen.add(passage);
      const id = item.version_id;
      const source = sources.get(id) || { id, title: item.source_title, chapters: [] };
      const chapterId = id + ":" + (item.passage_display?.chapter_key || item.chapter || "");
      const chapter = chapters.get(chapterId) || { id: chapterId, title: item.chapter, items: [], unknown: [] };
      if (!chapters.has(chapterId)) { chapters.set(chapterId, chapter); source.chapters.push(chapter); }
      chapter.items.push(item); sources.set(id, source);
    }
    for (const chapter of chapters.values()) {
      chapter.items.sort((a, b) => (a.passage_display?.order ?? Number.MAX_SAFE_INTEGER) - (b.passage_display?.order ?? Number.MAX_SAFE_INTEGER));
      if (chapter.items.some(item => item.passage_display?.title)) {
        chapter.unknown = chapter.items.filter(item => item.passage_display?.kind === "PARAGRAPH" && !item.passage_display.title);
        chapter.items = chapter.items.filter(item => !chapter.unknown.includes(item));
      }
    }
    return [...sources.values()];
  }
  function searchResultCards() {
    const sources = searchDocuments(), items = sources.flatMap(source => source.chapters.flatMap(chapter => [...chapter.items, ...chapter.unknown]));
    const passages = items.filter(item => item.passage_key).length, others = items.length - passages;
    const documents = new Set(items.map(item => item.document_id || item.version_id)).size;
    const count = state.submittedImageIntent
      ? documents ? `${documents} 份文档 · ${items.length} 条图片结果` : "未找到核验通过的图片"
      : documents ? `${documents} 份文档 · ${passages} 处相关段落${others ? ` · ${others} 条其他证据` : ""}` : "本页没有可展示的结果";
    const hit = (item, unknown = false) => `<button type="button" class="knowledge-search-hit" data-action="evidence" data-key="${esc(item.key)}" aria-pressed="${state.anchor === item.key}">
      <span>${unknown ? "" : `<strong>${esc(item.passage_display?.title || (item.passage_display?.kind === "SECTION" ? "章节标题" : item.passage_key ? "相关段落" : label(item.kind)))}</strong> `}<small>${esc(locationOf(item))}</small></span>
      <span>${esc(item.passage_display?.text || item.text || (unknown ? "查看摘录" : "查看证据"))}</span></button>`;
    return {
      count: `${count} · ${esc(label(state.result.status))}${state.result.reason ? " · " + esc(label(state.result.reason)) : ""}`,
      cards: sources.map((source, index) => `<details class="knowledge-search-source" data-search-fold="${esc("document:" + source.id)}"${index === 0 ? " open" : ""}>
        <summary class="knowledge-search-document"><strong>${esc(source.title || "来源文档")}</strong> <small>${source.chapters.reduce((count, chapter) => count + chapter.items.length + chapter.unknown.length, 0)} 处命中</small></summary>
        ${source.chapters.map(chapter => `<details class="knowledge-search-chapter" data-search-fold="${esc("chapter:" + chapter.id)}" open>
          <summary>${esc(chapter.title || "未归入章节")}</summary>
          ${chapter.items.map(item => hit(item)).join("")}
          ${chapter.unknown.length ? `<details class="knowledge-search-unknown" data-search-fold="${esc("unknown:" + chapter.id)}">
            <summary>另有 ${chapter.unknown.length} 处摘录未定位到条款</summary>${chapter.unknown.map(item => hit(item, true)).join("")}</details>` : ""}
        </details>`).join("")}</details>`).join(""),
    };
  }
  function scrollToSearchMatch() {
    const box = host.querySelector(".knowledge-search-context");
    const hit = box?.querySelector("[data-passage-anchor] mark") || box?.querySelector("[data-passage-anchor]");
    if (box && hit) box.scrollTop += hit.getBoundingClientRect().top - box.getBoundingClientRect().top - 20;
  }
  function passageHtml(anchor) {
    const data = state.passage;
    if (!data) return state.busy ? '<p role="status">正在读取原文上下文…</p>' :
      button("context-retry", "重试读取上下文");
    if (!data.blocks.length) return `<p>未保存可用的段落定位，以下仅为命中片段。</p><p class="knowledge-excerpt">${esc(anchor.text || "")}</p>`;
    return `<section class="knowledge-passage" aria-label="段落与上下文">
      ${data.has_before ? button("context-before", "展开前文") : ""}
      ${data.blocks.map(block => `<div class="knowledge-passage-block"${block.anchor ? ' data-passage-anchor=""' : ""}>
        <p class="knowledge-excerpt">${block.segments.map(part => part.matched ? `<mark>${esc(part.text)}</mark>` : esc(part.text)).join("")}</p>
        </div>`).join("")}
      ${data.has_after ? button("context-after", "展开后文") : ""}
      </section>`;
  }
  function readerPdf() {
    const { readerPage, readerZoom, passage } = state;
    return `<div class="knowledge-search-pdf"><div class="knowledge-search-pdf-tools">
      <form class="knowledge-pdf-page" data-form="pdf-page" aria-label="跳转 PDF 页码">
        <button type="button" class="icon-button" aria-label="上一页" data-action="pdf-step" data-key="-1"${readerPage <= 1 ? " disabled" : ""}>‹</button>
        <label>页码 ${input("pdfPage", readerPage, 'type="number" min="1" step="1" required')}</label>
        <small data-pdf-page-count hidden></small>${submit("跳转", false, true)}
        <button type="button" class="icon-button" aria-label="下一页" data-action="pdf-step" data-key="1" disabled>›</button>
      </form>
      ${readerPage !== passage.pdf_page ? button("pdf-hit", "回到命中页") : ""}
      <div class="knowledge-pdf-scale" role="group" aria-label="PDF 缩放">
        <button type="button" class="icon-button" aria-label="缩小" data-action="pdf-zoom" data-key="-25"${readerZoom === 50 ? " disabled" : ""}>−</button>
        <span>${readerZoom ? readerZoom + "%" : "适宽"}</span>
        <button type="button" class="icon-button" aria-label="放大" data-action="pdf-zoom" data-key="25"${readerZoom === 300 ? " disabled" : ""}>+</button>
        ${button("pdf-fit", "适合宽度")}
      </div>
    </div><div class="knowledge-pdf-viewport" data-pdf-viewport role="region" aria-label="PDF 原页" tabindex="0"><p role="status">正在加载 PDF 阅读器…</p></div></div>`;
  }
  function showPdfPageCount(count) {
    const form = pdfViewport?.parentElement?.querySelector('[data-form="pdf-page"]');
    if (!form) return;
    const field = form.elements.pdfPage;
    if (count) field.max = count; else field.removeAttribute("max");
    const label = form.querySelector("[data-pdf-page-count]");
    label.textContent = count ? "/ " + count : "";
    label.hidden = !count;
    form.querySelector('[data-action="pdf-step"][data-key="1"]').disabled = state.busy || !count || state.readerPage >= count;
  }
  function syncPdfReader() {
    const slot = (dialog.open ? dialog : host).querySelector("[data-pdf-viewport]");
    if (!slot) {
      pdfReader?.destroy(); pdfReader = pdfViewport = null;
      return;
    }
    if (pdfViewport && pdfViewport !== slot) slot.replaceWith(pdfViewport);
    else pdfViewport = slot;
    const viewport = pdfViewport;
    pdfModule ??= import(pdfReaderUrl).catch(error => { pdfModule = null; throw error; });
    void pdfModule.then(module => {
      if (viewport !== pdfViewport || !viewport.isConnected) return;
      pdfReader ??= module.mountPdfSearchReader(viewport, showPdfPageCount);
      pdfReader.update({ url: state.passagePdf, page: state.readerPage, zoom: state.readerZoom,
        hitPage: state.passage.pdf_page, blocks: state.passage.blocks });
      showPdfPageCount(pdfReader.pageCount);
    }).catch(() => {
      if (viewport === pdfViewport && viewport.isConnected)
        viewport.textContent = "PDF 阅读器未加载，可以切到文字上下文核对。";
    });
  }
  function readerHtml() {
    const anchor = evidence(state.anchor), groups = selectedGroups(anchor);
    const relatedKeys = [...new Set(groups.flatMap(value => value.linked_keys))];
    const tabs = state.passage?.pdf_page ? `<div class="knowledge-search-reader-tabs" role="group" aria-label="阅读方式">
      <button type="button" class="secondary-button" data-action="reader-tab" data-key="pdf" aria-pressed="${state.readerTab === "pdf"}">PDF 原文</button>
      <button type="button" class="secondary-button" data-action="reader-tab" data-key="context" aria-pressed="${state.readerTab === "context"}">${isInlineImage(anchor) ? "图片证据" : "文字上下文"}</button>
      </div>` : "";
    const content = !anchor ? `<p class="knowledge-search-empty">${searchWait ? "检索结果返回后，可在此阅读原文。" : "选择左侧一条结果，在这里核对原文。"}</p>` :
      state.readerTab === "pdf" ? (state.readerExpanded ? "" : state.passagePdf ? readerPdf() :
        state.busy ? '<p class="knowledge-search-empty" role="status">正在读取 PDF 原文…</p>' :
          `<div class="knowledge-search-empty">${button("context-pdf", "重试读取 PDF")}</div>`) :
      `<div class="knowledge-search-scroll knowledge-search-context" data-context-anchor="${esc(anchor.key)}">
        ${anchor.passage_key ? passageHtml(anchor) : `<p class="knowledge-excerpt">${esc(anchor.text || "")}</p>
          ${previewImage(anchor) || (canPreview(anchor) ? button("content", isInlineImage(anchor) ? "预览图片" : "读取内容", anchor.key)
            : "<p>此证据暂无可用的在线预览。</p>")}`}
        ${groups.length ? "<h3>关联证据</h3>" : ""}
        ${relatedKeys.map(key => {
          const related = evidence(key);
          return `<article class="knowledge-record"><strong>${esc(label(related?.kind))} · ${esc(locationOf(related))}</strong>
            <p class="knowledge-excerpt">${esc(related?.text || "")}</p>
            ${previewImage(related) || (canPreview(related) && !(isInlineText(related) && related.text !== null)
              ? button("content", isInlineImage(related) ? "预览关联图片" : "读取关联内容", key) : "")}</article>`;
        }).join("")}
        ${groups.map((related, index) => related.continuation
          ? button("relations", "继续读取关联" + (groups.length > 1 ? ` · 第 ${index + 1} 组` : ""), related.anchor_key) : "").join("")}
      </div>`;
    return `<section class="table-card knowledge-search-reader" aria-label="原文阅读" tabindex="-1">
      <header class="knowledge-search-panel-head"><div><h2>原文阅读</h2>
      ${anchor ? `<small>${esc([anchor.source_title, anchor.chapter, locationOf(anchor)].filter(Boolean).join(" · "))}</small>` : ""}</div>
      ${state.passagePdf ? button("reader-expand", "放大阅读区") : ""}</header>${tabs}${content}
    </section>`;
  }
  function renderSearch() {
    const exact = library()?.search_capabilities?.some(channel => channel.kind === "NEAR_DUPLICATE" && channel.direction === "IMAGE_TO_IMAGE");
    const related = state.mode === "RELEVANCE";
    const imageQuery = related && Boolean(state.queryImage);
    const onlyImages = imageQuery || state.output === "IMAGE";
    const result = state.result && searchResultCards();
    return `<div class="knowledge-search-workspace">
      <form class="table-card knowledge-search-formbar" data-form="search" novalidate>
        <div class="knowledge-search-top">${libraryPicker("知识库")}
          ${related ? state.queryImage ? `<div class="knowledge-query-image"><img class="knowledge-query-thumb" src="${esc(state.queryImageUrl)}" alt=""><span>${esc(state.queryImage.name)}</span>${button("remove-image", "移除图片")}</div>` :
            field("查询文字", input("query", state.query, 'type="search" maxlength="32000" required placeholder="输入要查的内容…"')) :
            `<p class="field-help">按标题或日期查找${state.mode === "EXHAUSTIVE" ? " · 逐页查全" : ""}</p>`}
          ${related && exact && !state.queryImage ? '<label for="knowledge-search-image" class="secondary-button knowledge-search-image-label">选择查询图</label><input id="knowledge-search-image" class="knowledge-search-image-input" type="file" name="queryFile" data-query-image aria-label="选择查询图" accept="image/png,image/jpeg,image/webp">' : ""}
          <button type="submit" class="primary-button" data-search-submit${searchWait || !library() ? " disabled" : ""}>${searchWait ? "检索中…" : "检索"}</button>
          <button type="button" class="secondary-button" data-action="search-options" data-key="" aria-expanded="${state.searchOptionsOpen}" aria-controls="knowledge-search-options">${state.searchOptionsOpen ? "收起条件" : "更多条件"}</button>
          ${button("search-history", "检索历史", "", "secondary-button", Boolean(searchWait))}
        </div>
        <div class="knowledge-search-scope">
          ${imageQuery ? "" : `<label class="knowledge-search-check"><input name="includeImages" type="checkbox" role="switch"${state.output === "ALL" ? " checked" : ""}> ${related ? "同时找相关图片" : "结果也包含图片"}</label>`}
          ${onlyImages ? `<small class="muted">${imageQuery ? "已选查询图 · " : "当前："}只找图片</small>` : ""}
          ${imageQuery ? `<label class="control-field">图片检索方式<select name="imageIntent"${searchWait ? " disabled" : ""}><option value="SOURCE"${state.imageIntent !== "SIMILAR" ? " selected" : ""}>找这张图的出处</option><option value="SIMILAR"${state.imageIntent === "SIMILAR" ? " selected" : ""}>找相似图（需视觉索引）</option></select></label>` : ""}
        </div>
        <div class="knowledge-search-options" id="knowledge-search-options"${state.searchOptionsOpen ? "" : " hidden"}>
          ${field("主结果数", input("count", state.count, 'type="number" min="1" max="100"'))}
          <label class="knowledge-search-check"><input name="onlyImages" type="checkbox"${onlyImages ? " checked" : ""}${imageQuery ? " disabled" : ""}> 只找图片</label>
          <label class="knowledge-search-check"><input name="structuredSearch" type="checkbox"${related ? "" : " checked"}> 按标题或日期查找</label>
          ${related ? "" : `${field("标题前缀", input("prefix", state.prefix, 'maxlength="256"'))}
            ${field("版本创建时间从（本地）", input("createdFrom", state.createdFrom, 'type="datetime-local"'))}
            ${field("版本创建时间早于（本地）", input("createdBefore", state.createdBefore, 'type="datetime-local"'))}
            ${field("排序字段", select("order", [["title","标题"],["created_at","版本创建时间"]], state.order))}
            <label class="knowledge-search-check"><input name="descending" type="checkbox"${state.descending ? " checked" : ""}> 降序</label>
            <label class="knowledge-search-check"><input name="exhaustiveSearch" type="checkbox"${state.mode === "EXHAUSTIVE" ? " checked" : ""}> 逐页查全</label>`}
        </div>
        ${!eligible().length ? "<small>当前没有可检索的已启用知识库。</small>" : ""}
      </form>
      <div class="knowledge-search-split">
        <section class="table-card knowledge-search-list" aria-label="检索结果">
          <header class="knowledge-search-panel-head" aria-live="polite"><h2>检索结果</h2>
            ${result ? `<p>${result.count}</p><small>${esc([state.summary, state.submittedSearchOptions].filter(Boolean).join(" · "))}</small>` :
              searchWait ? "<p>等待本次检索结果</p>" : !state.historyEntry ? "<p>选择知识库，输入文字后检索。</p>" : ""}
            ${state.historyEntry ? `<p>${esc(state.historyStatus)}</p>${button("history-rerun", "重新检索", "", "secondary-button", !library() || state.historyEntry.conditions.image_count > 1)}` : ""}</header>
          ${searchWait ? `<section class="knowledge-search-wait" aria-label="检索等待">
            <div class="knowledge-search-wait-copy"><p role="status">正在检索知识库…</p>
              <small>${esc([state.summary, state.submittedSearchOptions].filter(Boolean).join(" · "))}</small></div>
            <div class="knowledge-search-wait-actions"><span class="knowledge-search-wait-track" aria-hidden="true"></span>
              <span class="knowledge-search-wait-time" data-search-elapsed>已等待 ${searchSeconds(searchWait)} 秒</span>${button("stop-search", "停止等待")}</div>
            <p class="knowledge-search-wait-slow" data-search-slow role="status"${searchSeconds(searchWait) < slowSearchSeconds ? " hidden" : ""}>这次检索用时较长，可继续等待或停止等待。</p>
          </section>` : ""}
          ${result ? `<div class="knowledge-search-scroll" data-search-tree="${state.searchTreeRevision}">${result.cards}
            <div class="knowledge-search-tail">${state.result.continuation ? button("continue", "下一页主结果") : !state.historyEntry && state.submittedMode === "EXHAUSTIVE" &&
              state.result.scope_exhausted && !["PARTIAL","FAILED"].includes(state.result.status) ? "<p>当前条件下的授权范围已查完。</p>" : ""}</div>
            </div>` : ""}
        </section>
        ${readerHtml()}
      </div>
    </div>`;
  }
  function uploadSize(bytes) {
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + " MiB";
    if (bytes >= 1024) return Math.round(bytes / 1024) + " KiB";
    return bytes + " B";
  }
  function renderUpload() {
    const uploads = state.uploads, targetLibrary = library();
    const formats = Object.entries(targetLibrary?.upload_formats || {});
    const accept = formats.flatMap(([mime, ext]) => [mime, "." + ext]).join(",");
    const allowed = canUpload(targetLibrary);
    return `<form class="knowledge-upload-form" id="knowledge-file-upload" data-form="upload">
      <div class="knowledge-upload-destination">
        <div><small>目标知识库</small><strong>${esc(targetLibrary?.display_name)}</strong></div>
        <span class="status-pill ${allowed ? "status-success" : "status-muted"}">${allowed ? "可上传" : "只读"}</span>
      </div>
      <section class="knowledge-drop" aria-label="文件拖放区">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4m-5 5 5-5 5 5M4 16v4h16v-4"/></svg>
        <h3>拖放文件到这里</h3><p>可以同时选择多个文件</p>
        ${button("pick-upload", "选择文件", "", "secondary-button", !allowed)}
        <input type="file" multiple hidden data-upload accept="${esc(accept)}"${allowed ? "" : " disabled"}>
      </section>
      <div class="knowledge-upload-limits">
        <small>最多 20 个文件 · 单文件不超过 ${targetLibrary?.upload_max_bytes / 1048576} MiB</small>
        ${formats.length ? `<small>支持 ${esc(formats.map(([, ext]) => ext.toUpperCase()).join("、"))}</small>` : ""}
      </div>
      ${state.uploadError ? `<p class="table-error" role="alert">${esc(state.uploadError)}</p>` : ""}
      <section class="knowledge-upload-queue" aria-label="本次上传">
        <div class="knowledge-upload-queue-heading"><h3>本次上传</h3><small>${uploads.length} 个文件</small></div>
        ${uploads.length ? `<ul class="knowledge-upload-list">${uploads.map(({ file, key, mime, receipt, failure }) => `<li class="knowledge-upload-item">
          <span class="knowledge-upload-format" aria-hidden="true">${esc(targetLibrary?.upload_formats[mime]?.toUpperCase() || "文件")}</span>
          <div class="knowledge-upload-file"><strong>${esc(file.name)}</strong>
            <div class="knowledge-upload-meta"><small>${uploadSize(file.size)}</small>
              <span class="status-pill ${receipt ? "status-success" : failure ? "status-danger" : "status-muted"}">${receipt ? "已接收" : failure ? "接收失败" : "待提交"}</span>
            </div>${failure ? `<small class="knowledge-upload-failure">${esc(failure)}</small>` : ""}
          </div>${receipt ? `<a class="row-action" href="${esc(target("processing", receipt.version_id, receipt.job_id))}">查看解析任务</a>`
            : button("remove-upload", "移除文件", key, "row-action")}</li>`).join("")}</ul>` : `<p class="knowledge-upload-empty">还没有选中文件。</p>`}
      </section>
      <p class="knowledge-upload-note">每个文件单独提交和审核。文件接收后进入待审核，不会直接发布。</p>
    </form>`;
  }
  function renderUploadFooter() {
    const total = state.uploads.length, received = state.uploads.filter(entry => entry.receipt).length;
    const failed = state.uploads.filter(entry => !entry.receipt && entry.failure).length, complete = total > 0 && received === total;
    const summary = state.busy ? "正在逐份提交…" : received || failed ? `已接收 ${received} / ${total} 个文件`
      : total ? `已选 ${total} 个文件` : "尚未选择文件";
    return `<div class="knowledge-upload-summary" role="status"><strong>${summary}</strong>${failed ? `<small>${failed} 个文件需要重试</small>` : ""}</div>
      <div class="knowledge-upload-actions">
        ${button("close", complete ? "完成" : "关闭", "", complete ? "primary-button" : "secondary-button", state.busy)}
        ${complete ? "" : submit(state.busy ? "提交中…" : received || failed ? "提交未接收文件" : "提交文件",
          state.busy || !canUpload(library()) || !total, false, "knowledge-file-upload")}
      </div>`;
  }
  function renderDirectoryLibraries(libraries) {
    return `<aside class="table-card knowledge-sheet knowledge-directory-libraries" aria-label="选择知识库"><h2>选择知识库</h2>
      ${libraries.map(item => `<button type="button" class="knowledge-library-pick" data-action="pick-library" data-key="${esc(item.library_id)}" aria-pressed="${state.libraryId === item.library_id}"${state.busy ? " disabled" : ""}>
        <span class="knowledge-library-pick-heading"><strong>${esc(item.display_name)}</strong>${view === "files" ? `<span class="status-pill ${canUpload(item) ? "status-success" : "status-muted"}">${canUpload(item) ? "可上传" : "只读"}</span>` : ""}</span>
        ${item.description ? `<small class="knowledge-library-pick-description" title="${esc(item.description)}">${esc(item.description)}</small>` : ""}</button>`).join("")}
      ${!libraries.length && !state.busy ? "<p>当前工作域没有可访问的知识库。</p>" : ""}</aside>`;
  }
  function renderDirectoryFilters() {
    const files = view === "files", processing = view === "processing";
    const disabled = !library() || state.busy;
    const [title, scope] = files ? ["文件列表", "文件与版本"] : ["解析任务列表", "解析任务"];
    const statuses = files && !state.queue ? [["", "全部状态"], ["PUBLISHED", "已发布"], ["WITHDRAWN", "已撤回"]] : processing ?
      [["", "全部进度"], ...["AWAITING_APPROVAL","RECEIVED","PARSING","PARSED","READY","RETRY","FAILED","INDEX_FAILED","CANCELLED","REJECTED"].map(value => [value, taskLabel(value)])] : [];
    return `<header class="knowledge-directory-head">
      <div class="knowledge-directory-copy"><h2>${title}</h2><small>仅显示有权查看的${scope}。</small></div>
      <form class="knowledge-directory-filters${processing ? " knowledge-processing-filters" : ""}" role="search" aria-label="筛选${title}" data-form="filter">
        <label class="search-field">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6"/><path d="m15 15 5 5"/></svg>
          ${input("filterTitle", state.filterTitle, 'type="search" maxlength="256" placeholder="查找文件标题" aria-label="按文件标题查找"' + (disabled ? " disabled" : ""))}
        </label>
        <div class="knowledge-directory-filter-actions">
          ${statuses.length ? `<select name="filterState" aria-label="${files ? "发布状态" : "处理进度"}"${disabled ? " disabled" : ""}>${options(statuses, state.filterState)}</select>` : ""}
          ${processing && can(library(), "REVIEW", "MANAGE") ? `<label class="knowledge-directory-mine"><input name="mine" type="checkbox"${state.mine ? " checked" : ""}${disabled ? " disabled" : ""}> 仅本人</label>` : ""}
          ${submit("筛选", disabled, true)}
          ${processing ? button("refresh-progress", "刷新进度", "", "secondary-button", disabled) : ""}
        </div>
      </form>
    </header>`;
  }
  const selectedReviews = () => state.rows.filter(row => state.batchIds.has(row.id) && allowed(row, "review"));
  const awaitingApproval = row => row.parse_status === "AWAITING_APPROVAL";
  const commonStrategies = rows => Object.keys(strategyLabels).filter(strategy => rows.filter(awaitingApproval)
    .every(row => row.parse_strategy_options?.some(option => option.strategy === strategy && option.selectable)));
  function renderBatchReview() {
    const rows = selectedReviews(), needsStrategy = rows.some(awaitingApproval), strategies = commonStrategies(rows);
    return `<section class="knowledge-directory-queues" aria-label="批量审批">
      <span role="status">已选 ${rows.length} 个版本</span>
      ${needsStrategy ? `<select name="batchStrategy" aria-label="批量审批解析方案"${state.busy ? " disabled" : ""}>${emptyStrategyOption(state.batchStrategy)}${options(strategies.map(strategy => [strategy, strategyLabels[strategy]]), state.batchStrategy)}</select>` : ""}
      ${button("batch-approve", "批量通过", "", "primary-button", !rows.length || (needsStrategy && !strategies.includes(state.batchStrategy)))}
      ${button("batch-reject", "批量拒绝", "", "secondary-button", !rows.length)}
      ${needsStrategy && !strategies.length ? "<small>所选版本没有共同可用的解析方案，请分开审批。</small>" : ""}
    </section>`;
  }
  function renderDirectory() {
    const processing = view === "processing", files = view === "files", reviewing = files && Boolean(state.queue);
    const libraries = eligible();
    const queues = files ? reviewStates() : [], batch = reviewing && state.queue === "PENDING" && state.rows.some(row => allowed(row, "review"));
    const publicationColumn = reviewing ? "发布" : "状态";
    const columns = processing ? ["文件名", "版本创建时间", "处理进度", "解析结果"] :
      ["文件名", "版本创建时间", reviewing ? "处理阶段" : "格式", ...(reviewing ? ["审核"] : []), publicationColumn, "操作"];
    const cell = (name, html) => `<td data-label="${esc(name)}">${html}</td>`;
    const badge = (value, tone = "") => `<span class="status-pill ${tone}">${esc(value)}</span>`;
    return `<div class="knowledge-directory-layout">${renderDirectoryLibraries(libraries)}<section class="table-card knowledge-sheet">
      <header class="knowledge-current-library"><div><small>当前知识库</small><h2>${esc(library()?.display_name || "尚未选择")}</h2></div>${files ? button("add-file","＋ 上传资料","","primary-button", !canUpload(library())) :
        `<div class="knowledge-directory-filter-actions"><button type="button" class="secondary-button" data-action="progress-help" aria-haspopup="dialog"><span aria-hidden="true">ⓘ</span> 进度说明</button>
        ${canIndex("MANAGE") ? button("index-refresh", state.indexTasks ? "刷新索引任务" : "索引重建") : ""}</div>`}</header>
      ${processing && state.indexTasks ? renderIndexTasks() : ""}
      ${renderDirectoryFilters()}
      ${queues.length ? `<nav class="knowledge-directory-queues" aria-label="资料队列">${["", ...queues].map(value =>
        `<button type="button" class="secondary-button" data-action="queue" data-key="${value}" aria-pressed="${state.queue === value}"${state.busy ? " disabled" : ""}>${queueLabel(value)}</button>`).join("")}</nav>` : ""}
      ${batch ? renderBatchReview() : ""}
      ${state.batchFailures.length ? `<ul class="table-error" aria-label="批量审批失败的版本">${state.batchFailures.map(message => `<li>${esc(message)}</li>`).join("")}</ul>` : ""}
      <div class="table-scroll"><table class="knowledge-directory-table"><thead><tr>${batch ? "<th>批选</th>" : ""}${columns.map(name => `<th>${name}</th>`).join("")}</tr></thead><tbody>
      ${state.rows.map(row => `<tr>
        ${batch ? `<td>${allowed(row, "review") ? `<label class="knowledge-directory-mine"><input type="checkbox" name="batchVersion" value="${esc(row.id)}" aria-label="选择文件 ${esc(row.title)} · ${esc(versionTime(row.created_at))}"${state.batchIds.has(row.id) ? " checked" : ""}${state.busy ? " disabled" : ""}></label>` : ""}</td>` : ""}
        ${cell(columns[0], `<strong>${esc(row.title)}</strong>`)}
        ${cell(columns[1], esc(versionTime(row.created_at)))}
        ${!processing ? cell(columns[2], reviewing ? badge(taskLabel(row.processing_status),
          row.error_code ? "status-warning" : row.generation_ready ? "status-success" : "") +
          (row.error_code ? `<small>错误码：${esc(row.error_code)}</small>` : "") : esc(row.mime_type)) : ""}
        ${reviewing ? cell("审核", badge(label(row.review), row.review === "APPROVED" ? "status-success" : row.review === "PENDING" ? "status-warning" : "")) : ""}
        ${!processing ? cell(publicationColumn, !reviewing && pendingFileReview(row) ? badge("待审核", "status-warning") :
          badge(publication(row.status), row.status === "PUBLISHED" ? "status-success" : "status-muted")) : ""}
        ${processing ? cell("处理进度", renderProgress(row)) : ""}
        ${processing ? cell("解析结果", `<button type="button" class="text-button knowledge-result-action" aria-haspopup="dialog" aria-label="查看处理结果：${taskResult(row)}" data-parse-status="${esc(row.parse_status)}"${row.processing_status === "INDEX_FAILED" ? " data-index-failed" : ""} data-action="row" data-key="${esc(row.id)}"${state.busy ? " disabled" : ""}>${taskResult(row)}<span aria-hidden="true">›</span></button>`) :
        `<td><div class="knowledge-directory-row-actions">${button("row", "查看文件", row.id, "row-action")}
          ${files && row.reviewTarget ? button("govern", row.reviewTarget === row.id ? "审批" : "审批其他版本", row.reviewTarget, "row-action") : ""}
          ${reviewing && allowed(row, "withdraw") ? button("govern", "撤回", row.id, "row-action") : ""}</div></td>`}
      </tr>`).join("")}</tbody></table></div>
      ${!state.rows.length ? `<p>${!library() ? (libraries.length ? "请先选择知识库。" : "当前没有可用知识库。") : state.filterTitle || state.filterState || state.queue ? "没有匹配当前筛选的文件。" : "当前没有可见文件。"}</p>` : ""}
      ${state.rows.length || state.page > 1 ? `<footer class="section-toolbar">${button("previous", "上一页", "", "secondary-button", state.page <= 1)}
        <span>第 ${state.page} 页</span>${button("next", "下一页", "", "secondary-button", !state.more)}</footer>` : ""}
    </section></div>`;
  }
  function renderIndexTasks() {
    const data = state.indexTasks, batch = data.batch;
    const terminal = !activeIndexBatch(batch);
    return `<section class="drawer-form-section" aria-label="索引重建任务"><h3>索引重建任务</h3>
      ${data.pending_roles.length ? `<p>待重建：${esc(data.pending_roles.map(role => indexNames[role]).join("、"))} · ${data.published_file_count} 个已发布文件</p>` : ""}
      ${data.active_file_rebuild_count > 0 ? `<p>有 ${data.active_file_rebuild_count} 个单文件重建任务未结束，暂不能开始整库重建。</p>` : ""}
      ${terminal && data.pending_roles.length && canIndex("MANAGE") ? button("index-start", "开始重建", "", "secondary-button", data.active_file_rebuild_count > 0) : ""}
      ${batch ? `<p>${esc(indexBatchLabel(batch))} · ${esc(batch.roles.map(role => indexNames[role]).join("、") || "基础索引")}
        · ${batch.items.filter(item => item.status === "READY").length} / ${batch.items.length} 个文件就绪</p>
        ${!terminal ? `<ul>${batch.items.map(item => `<li><strong>${esc(item.title)}</strong>
          ${item.index_progress ? renderRebuildProgress(item) : ` · ${esc(rebuildLabels[item.status])}`}
          ${item.visual_not_applicable ? " · 无图片，视觉索引不适用" : ""}
          ${item.status === "FAILED" && batch.status !== "STALE" && canIndex("MANAGE") ? button("index-retry", "重试此文件", item.version_id, "row-action") : ""}</li>`).join("")}</ul>` : ""}
        <div class="section-toolbar">
          ${!terminal && canIndex("MANAGE") ? button("index-discard", "放弃批次") : ""}
        </div>` : "<p>当前没有索引重建批次。</p>"}
    </section>`;
  }
  function fileIndexDetail(row) {
    const batch = state.detailIndex?.batch, active = activeIndexBatch(batch);
    const item = batch?.items.find(entry => entry.version_id === row.id);
    if (!allowed(row, "rebuild-index") && !item) return { body: "", secondary: "", primary: "" };
    const status = [
      item ? `<p>${esc(indexBatchLabel(batch))} · ${esc(rebuildLabels[item.status])}
        ${item.error_code ? " · 错误码：" + esc(item.error_code) : ""}</p>${active ? renderRebuildProgress(item) + renderBatchDetail(item.index_progress, "重建批次明细") : ""}` : "",
      active && batch.scope === "LIBRARY" ? "<p>当前库正在进行整库重建，可在解析任务页查看批次。</p>" : "",
      allowed(row, "rebuild-index") && state.detailIndex && !active ? "<p>提交后在后台排队，复用现有解析结果。校验通过后自动生效。</p>" : "",
    ].join("");
    const secondary = (active || !state.detailIndex ? button("file-index-refresh", "刷新索引任务") : "") +
      (item && batch.scope === "FILE" && active && canIndex("MANAGE") ? button("file-index-discard", "放弃批次") : "");
    const primary = (allowed(row,"rebuild-index") && state.detailIndex && !active ? button("file-index-start", "重建索引", row.id, "primary-button") : "") +
      (item && active && item.status === "FAILED" && batch.status !== "STALE" && canIndex("MANAGE") ? button("file-index-retry", "重试重建索引", row.id, "primary-button") : "");
    return { body: status ? `<section class="drawer-form-section knowledge-task-section"><h3>索引重建</h3>${status}</section>` : "", secondary, primary };
  }
  function renderProcessingDetail(row) {
    const index = fileIndexDetail(row);
    const summary = facts([
      ["解析结果", parseResult(row)], ["索引", indexStage(row)],
      ["创建时间", versionTime(row.created_at)], ["审核", label(row.review)], ["发布", publication(row.status)],
    ], true);
    const diagnostic = facts([
      ["任务编号", row.job_id], ...(row.error_code ? [["处理错误码", row.error_code]] : []),
      ...(row.comment ? [["审核意见", row.comment]] : []),
    ]);
    return {
      body: `<section class="drawer-form-section knowledge-task-section"><h3>任务概况</h3>${summary}
        ${activeIndexBatch(state.detailIndex?.batch) && state.detailIndex.batch.items.some(item => item.version_id === row.id) ? "" :
          renderProgress(row) + renderBatchDetail(row.index_progress, row.progress_stage?.startsWith("REBUILD") ? "重建批次明细" : "批次明细")}
        <p class="field-help">解析提取文件内容，索引让这些内容可以被搜到。索引校验通过后自动发布或更新。</p>
        ${maintenanceHint(row) ? `<p>${maintenanceHint(row)}</p>` : ""}</section>
        ${allowed(row,"reprocess") ? `<section class="drawer-form-section knowledge-task-section"><h3>重新解析设置</h3>
        ${field("解析方式", `<select name="reprocessMode"><option value="ORIGINAL"${row.original_parse_available ? "" : " disabled"}${state.reprocessMode === "ORIGINAL" ? " selected" : ""}>沿用原解析方案</option>
          <option value="CURRENT"${state.reprocessMode === "CURRENT" ? " selected" : ""}>使用当前库模型重新解析</option></select>`)}
        ${row.original_parse_available ? "" : '<p class="field-help">原解析档位或模型版本不可用。</p>'}
        ${state.reprocessMode === "CURRENT" ? field("解析方案", strategyPicker(row)) : ""}
        <p class="field-help">${row.status === "PUBLISHED" ? "重新解析期间旧版仍可检索；校验通过后自动切换，失败不会影响旧版。" : "使用同一原文件并保留原任务记录。索引校验通过后自动发布。"}</p></section>` : ""}
        ${index.body}
        <section class="drawer-form-section knowledge-task-section"><h3>任务详情</h3>${diagnostic}
        ${row.parse_task_history?.length ? `<details><summary>历史解析任务（最近 ${row.parse_task_history.length} 条）</summary>
          <ul>${row.parse_task_history.map(task => `<li>${esc(versionTime(task.created_at))} · ${esc(task.status === "FAILED" ? "解析失败" : label(task.status))} · ${esc(task.job_id)}${task.error_code ? " · " + esc(task.error_code) : ""}</li>`).join("")}</ul></details>` : ""}
        </section>`,
      footer: href(target("files", row.id), "查看文件") + index.secondary +
        (allowed(row,"cancel") ? button("cancel","取消任务",row.id,"danger-button") : "") +
        (allowed(row,"reprocess") ? button("reprocess","重新解析",row.id,allowed(row,"rebuild-index") ? "secondary-button" : "primary-button", !canReprocess(row)) : "") +
        (allowed(row,"retry-index") ? button("retry-index","重试建索引",row.id,"primary-button") : "") + index.primary,
    };
  }
  function renderManagement() {
    const matches = matchingLibraries();
    return `<section class="resource-list-surface">
      <header class="resource-list-header">
        <div class="resource-list-heading"><h2>知识库目录</h2><p>创建和维护当前工作域的知识库。</p></div>
        <div class="resource-list-tools">
          <label class="search-field"><span aria-hidden="true">⌕</span>${input("filterTitle", state.filterTitle, 'type="search" placeholder="搜索知识库名称或说明" aria-label="搜索知识库"')}</label>
          <label class="resource-list-filter">库状态${select("filterState", [["","全部"],["ACTIVE","启用"],["DISABLED","停用"]], state.filterState)}</label>
          <span class="resource-list-count" data-catalog-count aria-live="polite">${matches.length} 项</span>
          ${button("reload", state.catalogLoading ? "刷新中…" : "刷新", "", "secondary-button", state.busy || state.catalogLoading)}
        </div>
      </header>
      <div class="resource-list-body" data-catalog-rows aria-busy="${state.catalogLoading}">${renderCatalogRows(matches)}</div>
    </section>`;
  }
  function matchingLibraries() {
    const keyword = state.filterTitle.trim().toLocaleLowerCase();
    return eligible().filter(item => (!state.filterState || state.filterState === item.status) &&
      [item.display_name, item.description || "", item.library_id].some(value => value.toLocaleLowerCase().includes(keyword)));
  }
  function renderCatalogRows(matches) {
    if (state.catalogLoading) return '<p class="empty-cell" role="status">正在读取知识库…</p>';
    return `<div class="management-catalog-grid">
      <button class="management-catalog-add-card" type="button" data-action="create" data-key=""><span class="management-catalog-add-mark" aria-hidden="true">+</span><strong>新建知识库</strong><small>为知识资料创建独立的知识空间</small></button>
      ${matches.map(item => `<article class="management-catalog-card knowledge-base-card${item.status === "ACTIVE" ? "" : " is-disabled"}">
        <header><div><p class="eyebrow">知识库</p><h3 title="${esc(item.display_name)}">${esc(item.display_name)}</h3></div><span class="status-pill ${item.status === "ACTIVE" ? "status-success" : "status-muted"}">${item.status === "ACTIVE" ? "启用" : "停用"}</span></header>
        <div class="aiops-catalog-card-body">
          <p class="knowledge-base-description" title="${esc(item.description || "暂无说明")}">${esc(item.description || "暂无说明")}</p>
          <small class="technical-meta knowledge-base-id" title="${esc(item.library_id)}">${esc(item.library_id)}</small>
          <dl><div><dt>授权成员</dt><dd>${item.grants ? Object.keys(item.grants).length + " 人" : "—"}</dd></div><div><dt>当前修订</dt><dd>${esc(item.row_version)}</dd></div></dl>
          <p class="knowledge-base-updated">更新于 <time datetime="${esc(item.updated_at)}">${esc(versionTime(item.updated_at))}</time></p>
        </div>
        <footer class="management-catalog-actions management-catalog-action-bar">
          <div class="management-catalog-primary-actions">${button("edit","编辑",item.library_id,"row-action")}${button("members","成员",item.library_id,"row-action")}</div>
          <div class="management-catalog-overflow">
            <button type="button" class="row-action management-catalog-more" data-action="catalog-more" data-key="${esc(item.library_id)}" aria-haspopup="menu" aria-expanded="false"><span>更多</span><span aria-hidden="true">⌄</span></button>
            <div class="management-catalog-menu" role="menu" hidden>
              <button type="button" class="row-action" role="menuitem" data-action="toggle" data-key="${esc(item.library_id)}">${item.status === "ACTIVE" ? "停用" : "启用"}</button>
              <button type="button" class="danger-row-action" role="menuitem" data-action="delete" data-key="${esc(item.library_id)}">删除</button>
            </div>
          </div>
        </footer>
      </article>`).join("")}
    </div>${!matches.length && !state.error ? `<p class="empty-cell" role="status">${eligible().length ? "没有匹配的知识库，请调整筛选条件。" : "当前工作域还没有可管理的知识库。"}</p>` : ""}`;
  }
  function filterCatalog() {
    const matches = matchingLibraries();
    host.querySelector("[data-catalog-count]").textContent = `${matches.length} 项`;
    host.querySelector("[data-catalog-rows]").innerHTML = renderCatalogRows(matches);
  }
  function closeCatalogMenus(restoreFocus = false) {
    host?.querySelectorAll(".management-catalog-menu:not([hidden])").forEach(menu => {
      const opener = menu.previousElementSibling;
      if (restoreFocus && menu.contains(document.activeElement)) opener.focus();
      menu.hidden = true;
      opener.setAttribute("aria-expanded", "false");
    });
  }
  function allowed(row, operation) {
    return row.allowed_actions?.includes(operation) && appCan(operation === "cancel" ? "upload" : ["review","reprocess"].includes(operation) ? "review" : "knowledge_core_manage");
  }
  function renderMembers(item) {
    const m = state.memberMatrix;
    const { ids, page, pages, count } = matrix.matrixPage(m);
    const selected = matrix.selected(m);
    const warning = matrix.memberWarning(m);
    const tabs = m.options ? `<div class="member-matrix-tabs" role="group" aria-label="成员筛选">${[
      ["members", "已授权 " + Object.keys(m.draft).length], ["candidates", "可添加"],
    ].map(([id, name]) => `<button class="secondary-button" type="button" aria-pressed="${m.tab === id}" data-action="member-tab" data-key="${id}">${name}</button>`).join("")}</div>` : "";
    const batch = selected.size && !m.blocked ? `<div class="member-matrix-note member-matrix-batch"><strong>已选 ${selected.size} 人</strong>
      ${m.tab === "members" ? `<select name="memberBatchOp" aria-label="批量权限">${options(matrix.batchOps.map(op => [op, matrix.memberLabels[op]]), m.batchOp)}</select>
        ${button("member-grant", "追加权限")}${button("member-revoke", "撤销权限")}` :
        button("member-batch-add", "批量加入（仅读取）", "", "primary-button")}
      ${button("member-clear", "清空选择")}</div>` : "";
    const rows = ids.map(id => {
      const candidate = matrix.member(m, id);
      const full = esc(`${matrix.personName(m, id)}（${id}）`);
      const ops = m.draft[id], protect = matrix.protectsManager(m, id);
      const controls = matrix.present(m, id) ?
        `<td><span class="member-matrix-read" aria-label="已授予读取（随成员生效）" title="读取随成员生效">✓</span></td>
          ${matrix.memberOps.slice(1).map(op => {
            const granted = ops.includes(op);
            const title = op === "MANAGE" ? (protect ? "最后一名有效管理员" : !candidate?.can_manage ? "无 App 管理资格" : "") : "";
            return `<td>${candidate ? `<input type="checkbox" data-member="${esc(id)}" value="${op}" aria-label="${matrix.memberLabels[op]} ${full}"
              ${title ? `title="${title}"` : ""}${granted ? " checked" : ""}${m.blocked || (op === "MANAGE" && (protect || (!candidate.can_manage && !granted))) ? " disabled" : ""}>` :
              `<span aria-label="${granted ? "已记录授权" : "未授权"}">${granted ? "✓" : "—"}</span>`}</td>`;
          }).join("")}
          <td>${m.options ? `<button class="danger-row-action row-action" type="button" data-action="remove-member" data-key="${esc(id)}" aria-label="移除 ${full}"
             ${protect ? 'title="最后一名有效管理员"' : ""}${m.blocked || protect ? " disabled" : ""}>移除</button>` : "—"}</td>` :
        `<td colspan="${matrix.memberOps.length}">尚未授权，加入后可设置</td><td><button class="row-action" type="button" data-action="add-member" data-key="${esc(id)}"
            aria-label="加入 ${full}"${m.blocked ? " disabled" : ""}>加入</button></td>`;
      return `<tr${selected.has(id) ? ' class="is-selected"' : ""}>
        <td>${candidate ? `<input type="checkbox" data-member-pick="${esc(id)}" aria-label="选择 ${full}"${selected.has(id) ? " checked" : ""}${m.blocked ? " disabled" : ""}>` : "—"}</td>
        <th scope="row"><strong>${esc(matrix.personName(m, id))}</strong><small>成员 ID：${esc(id)}</small>
          ${m.options && matrix.invalidGrant(m, id) ? `<small class="member-matrix-invalid">${candidate ? "管理资格已失效" : "成员资格已失效"}</small>` : ""}</th>${controls}</tr>`;
    }).join("");
    const accessHint = !m.options || m.blocked ? `<div class="section-toolbar"><p>${m.blocked ? "草稿已保留。读取最新授权后，需要重新复核。" :
      state.busy ? "正在读取授权和成员目录…" : "暂时无法验证候选成员，现有授权只读。"}</p>
      ${state.busy ? "" : button("reload-members", m.blocked ? "读取最新授权" : "重试读取")}</div>` : "";
    const body = `<h3>${esc(item.display_name)}</h3>
      <p class="member-matrix-note"><strong>仅对当前库生效。</strong> 加入即有读取授权；上传、审核等操作还需要相应的 App 权限。</p>
      ${m.note ? `<p role="status">${esc(m.note)}</p>` : ""}${accessHint}
      <div class="member-matrix-tools">${tabs}${input("memberQuery", m.query, 'type="search" aria-label="搜索姓名或成员 ID" placeholder="搜索姓名或成员 ID"')}</div>
      ${batch}<div class="member-matrix-frame" tabindex="0" role="region" aria-label="权限矩阵，可横向滚动">
        <table class="member-matrix" aria-label="知识库权限矩阵"><thead><tr><th scope="col">选</th><th scope="col">成员</th>
          ${matrix.memberOps.map(op => `<th scope="col">${matrix.memberLabels[op]}</th>`).join("")}<th scope="col">操作</th></tr></thead>
          <tbody>${rows || `<tr><td colspan="${matrix.memberOps.length + 3}">没有匹配的成员。</td></tr>`}</tbody></table>
      </div>
      <div class="member-matrix-tools"><small>匹配 ${count} 人 · 第 ${page} / ${pages} 页</small><div class="member-matrix-tabs">
        ${button("member-prev", "上一页", "", "secondary-button", page <= 1)}${button("member-next", "下一页", "", "secondary-button", page >= pages)}
      </div></div>
      <small>读取随成员生效；管理不包含其他列，且至少保留一名具备 App 管理资格的库管理员。勾选只改草稿。</small>
      ${warning ? `<p class="table-error" role="alert">${esc(warning)}</p>` : ""}`;
    const footer = `<span class="member-matrix-count" role="status">${matrix.changes(m).length} 人有改动 · 尚未保存</span>` +
      button("close", "取消") + button("save-members", "复核改动", "", "primary-button", !memberReady() || !membersDirty());
    return { body, footer };
  }
  function renderDrawer() {
    let title, body, footer = "";
    const row = state.row, item = state.managed, anchor = evidence(state.anchor);
    const documentDetail = ["files", "processing"].includes(view) && Boolean(row) && !state.uploadOpen;
    if (state.historyOpen) {
      title = "检索历史";
      body = `<p class="field-help">仅自己可见，保留最近 30 天。打开历史不会重新检索。</p>
        ${state.historyItems.length ? `<ul class="knowledge-history-list">${state.historyItems.map(entry =>
          `<li><div><strong>${esc(historyTitle(entry.conditions))}</strong><small>${esc([entry.library_name || "知识库不可访问", historyTime(entry.created_at), `${entry.result_count} 条主结果`].join(" · "))}</small></div>
          <div class="knowledge-history-actions">${button("history-view", state.historyId === entry.history_id ? "查看当前记录" : "查看", entry.history_id)}
          ${button("history-delete", "删除", entry.history_id, "secondary-button")}</div></li>`).join("")}</ul>` :
          `<p role="status">${state.busy ? "正在读取历史…" : "最近 30 天还没有检索记录。"}</p>`}`;
      if (state.historyMore) footer = button("history-more", "加载更多");
    } else if (state.progressHelp) {
      title = "进度说明";
      body = `<section class="drawer-form-section knowledge-task-section"><h3>识别与整理</h3><p>显示当前处理步骤，不估算时长或百分比。</p></section>
        <section class="drawer-form-section knowledge-task-section"><h3>分批建索引</h3><p>百分比按已提交批次 / 总批次计算。总数还没有确定时，只显示已提交批次数；重试同一任务时保留已提交批次。</p></section>
        <section class="drawer-form-section knowledge-task-section"><h3>何时显示 100%</h3><p>全部批次提交且最终校验通过后才显示 100%。首次处理自动发布；重建索引校验通过后自动生效。</p></section>`;
    } else if (state.uploadOpen) {
      title = "上传资料"; body = renderUpload();
      footer = renderUploadFooter();
    } else if (state.readerExpanded && anchor && state.passagePdf) {
      title = esc(anchor.source_title || "PDF 原文"); body = readerPdf();
    } else if (row && view === "processing") {
      title = "解析结果";
      ({ body, footer } = renderProcessingDetail(row));
    } else if (row) {
      title = state.governing ? "资料审批" : "查看文件";
      body = `${!state.governing && state.fileVersions.length > 1 ? field("可读版本", select("fileVersion", state.fileVersions.map(version =>
          [version.id, `${versionTime(version.created_at)} · ${publication(version.status)} · ${version.id}`]), row.id)) : ""}
        ${facts([
        ["版本",row.id],["创建时间",versionTime(row.created_at)],["处理",taskLabel(row.processing_status)],
        ["索引",indexStage(row)],["审核",label(row.review)],["发布",publication(row.status)],
        ...(row.error_code ? [["处理错误码",row.error_code]] : []),
        ...(row.comment ? [["审核意见",row.comment]] : []),
      ], documentDetail)}${filePreviewHtml(row)}`;
      if (state.governing) {
        if (allowed(row, "review")) body += `<section class="drawer-form-section">
        ${awaitingApproval(row) ? field("批准时选择解析方案",
          `<select name="parseStrategy">${emptyStrategyOption(state.parseStrategy)}${(row.parse_strategy_options || []).map(option => {
            const reason = option.executor_unavailable ? "模型解析执行端未启用" : (option.unavailable_roles || []).map(role => role.toUpperCase()).join("、") + " 不可用";
            return `<option value="${esc(option.strategy)}"${option.selectable ? "" : " disabled"}${state.parseStrategy === option.strategy ? " selected" : ""}>${esc(strategyLabels[option.strategy] + (option.selectable ? "" : `（${reason}）`))}</option>`;
          }).join("")}</select>`) + '<p class="field-help">审核通过后开始解析，索引校验通过后自动发布。</p>' : ""}
        ${field("审核意见（选填）", `<textarea name="comment" rows="4" maxlength="2000">${esc(state.comment)}</textarea>`)}
        </section>`;
        footer = (allowed(row,"review") ? button("approve","通过审核",row.id,"primary-button", !canApprove(row)) + button("reject","拒绝",row.id) : "") +
          (allowed(row,"withdraw") ? button("withdraw","撤回此版本",row.id,"danger-button") : "");
      }
    } else if (state.edit) {
      title = { create: "新建知识库", edit: "编辑知识库", members: "成员授权" }[state.edit];
      if (["create","edit"].includes(state.edit)) {
        const reviewing = state.edit === "create" && state.confirmingCreate;
        const fields = reviewing ? globalThis.KBotKnowledgeCapabilitiesReview.render(state.name.trim(),
          { embedding: state.embedding, visual_embedding: state.visual_embedding },
          { ocr: state.ocr, vlm: state.vlm }, state.modelCatalog, esc) : `<section class="drawer-form-section">
          <header><span>01</span><div><strong>基本信息</strong><small>名称在当前工作域内不可重复。</small></div></header>
          <label class="control-field form-span"><span class="field-label">知识库名称</span>${input("name", state.name, 'maxlength="256" required autofocus placeholder="例如：产品资料知识库"')}</label>
          <label class="control-field form-span"><span class="field-label">说明</span><textarea name="description" rows="5" maxlength="1000" placeholder="说明知识范围或适用团队">${esc(state.description)}</textarea></label>
        </section><section class="drawer-form-section"><header><span>02</span><div><strong>解析模型</strong><small>仅绑定本库模型，审批资料时才选择解析方案。</small></div>
          <button type="button" class="icon-button" data-knowledge-parse-help aria-haspopup="dialog" aria-expanded="false" aria-label="查看解析方案说明">?</button></header>
          ${modelFields(parseRoles)}
        </section><section class="drawer-form-section"><header><span>03</span><div><strong>索引与检索模型</strong><small>不配置文本 Embedding，新资料不支持文本语义检索，仍可按关键词检索。模型不影响解析选档，变更也不会改写旧索引。</small></div></header>
          ${modelFields(indexRoles)}
          ${state.edit === "edit" && item?.status === "ACTIVE" ? href(`./processing.html?library_id=${enc(item.library_id)}&index_rebuild=1`, "查看索引重建任务") : ""}
        </section>`;
        body = state.indexImpact ? `<section class="drawer-form-section"><h3>确认向量模型变更</h3>
          <ul>${changedIndexes().map(([role, , name]) => `<li>${esc(name)}：${state[role] ? item.index_models?.[role] ? "更换模型" : "新增模型" : "解绑，保存后立即关闭对应检索通道"}</li>`).join("")}</ul>
          <p>当前有 ${state.indexImpact.published_file_count} 个已发布文件。重建复用已有解析结果，整批校验通过后自动切换。</p>
          <p>仅保存不会重建已有文件；可稍后在解析任务中开始重建。关键词检索及原图匹配不受解绑影响。</p>
          ${activeIndexBatch(state.indexImpact.batch) ? "<p>已有未结束的整库重建任务。开始新批次前，请在解析任务页处理当前批次。</p>" : ""}
          ${state.indexImpact.active_file_rebuild_count > 0 ? `<p>有 ${state.indexImpact.active_file_rebuild_count} 个单文件重建任务未结束，暂不能开始整库重建。</p>` : ""}</section>`
          : `<form id="knowledge-library-editor" data-form="save">${fields}</form>`;

        footer = (reviewing ? button("back-create", "上一步") : button("close", "取消")) +
          (state.indexImpact ? button("index-back", "返回修改") + button("index-save", "仅保存") +
            (item?.status === "ACTIVE" && state.indexImpact.published_file_count > 0 && !activeIndexBatch(state.indexImpact.batch)
              && !state.indexImpact.active_file_rebuild_count
              && changedIndexes().some(([role]) => state[role])
              ? button("index-save-rebuild", "保存并重建", "", "primary-button") : "") :
            submit(state.edit === "edit" ? "保存知识库" : reviewing ? "确认创建" : "下一步", false, false, "knowledge-library-editor"));
      } else if (state.edit === "members") ({ body, footer } = renderMembers(item));
    }
    if (!title) {
      if (dialog.open) { dialog.close(); dialog.replaceChildren(); trigger?.focus(); }
      trigger = null;
      return;
    }
    const wasOpen = dialog.open;
    const wasReviewing = Boolean(dialog.querySelector("[data-knowledge-capability-title]"));
    dialog.classList.toggle("knowledge-editor-drawer", Boolean(state.edit));
    dialog.classList.toggle("knowledge-member-drawer", state.edit === "members");
    dialog.classList.toggle("knowledge-upload-drawer", state.uploadOpen);
    dialog.classList.toggle("knowledge-file-detail-drawer", documentDetail);
    dialog.classList.toggle("knowledge-search-expanded", Boolean(state.readerExpanded));
    dialog.classList.toggle("admin-drawer-sectioned", Boolean(state.edit) || state.uploadOpen || documentDetail);
    const subtitle = state.edit === "members" ? "管理当前工作域成员在此知识库的权限。"
      : state.edit === "create" ? (state.confirmingCreate ? "第 2 步，共 2 步 · 确认检索能力" : "第 1 步，共 2 步 · 填写信息与选择模型")
        : "修改知识库的基本信息与模型。";
    const heading = state.edit ? `<div><p class="eyebrow">知识库配置</p><h2 id="knowledge-drawer-title">${title}</h2><p>${subtitle}</p></div>
      <button type="button" class="icon-button" data-action="close" aria-label="关闭">×</button>` : state.uploadOpen || documentDetail ?
      `<div><p class="eyebrow">${documentDetail ? title : "知识库文件"}</p><h2 id="knowledge-drawer-title">${documentDetail ? esc(row.title) : title}</h2></div>
        <button type="button" class="icon-button" data-action="close" aria-label="关闭">×</button>` :
      `<h2 id="knowledge-drawer-title">${title}</h2>${button("close","关闭")}`;
    dialog.innerHTML = `<header class="admin-drawer-header">${heading}</header>
      <div class="admin-drawer-body">${body}${!state.historyOpen && state.preview ? `<img class="knowledge-preview" src="${esc(state.preview)}" alt="本次读取的原图">` : ""}${errorHtml()}</div>
      ${footer ? `<footer class="admin-drawer-footer">${footer}</footer>` : ""}`;
    if (!wasOpen) dialog.showModal();
    else if (state.edit === "create" && state.confirmingCreate !== wasReviewing) {
      dialog.querySelector(state.confirmingCreate ? "[data-knowledge-capability-title]" : '[form="knowledge-library-editor"]').focus();
    }
    else if (!dialog.contains(document.activeElement)) dialog.querySelector('[data-action="close"]').focus({ preventScroll: true });
  }
  function render() {
    if (!host) return;
    const active = document.activeElement;
    const action = active?.dataset?.action, key = active?.dataset?.key;
    const name = active?.name, insideDrawer = dialog.contains(active);
    const memberCell = active?.dataset?.member, memberPick = active?.dataset?.memberPick, operation = active?.value;
    const cursor = ["memberQuery", "query", "prefix"].includes(name) ? [active.selectionStart, active.selectionEnd] : null;
    const matrixFrame = dialog.querySelector(".member-matrix-frame");
    const matrixTop = matrixFrame && dialog.querySelector(".admin-drawer-body").scrollTop;
    const matrixLeft = matrixFrame?.scrollLeft;
    const previousTree = host.querySelector("[data-search-tree]");
    const sameTree = previousTree?.dataset.searchTree === String(state.searchTreeRevision);
    const folds = new Map(sameTree ? [...previousTree.querySelectorAll("details[data-search-fold]")].map(node => [node.dataset.searchFold, node.open]) : []);
    const previousContext = host.querySelector("[data-context-anchor]");
    const focusBack = trigger?.dataset?.action ? { action: trigger.dataset.action, key: trigger.dataset.key } : null;
    host.innerHTML = errorHtml() + (state.notice ? `<p role="status">${esc(state.notice)}</p>` : "") +
      (view === "search" ? renderSearch() : management ? renderManagement() : renderDirectory());
    for (const node of host.querySelectorAll("details[data-search-fold]")) {
      if (folds.has(node.dataset.searchFold)) node.open = folds.get(node.dataset.searchFold);
    }
    const resultScroll = host.querySelector("[data-search-tree]");
    if (resultScroll && sameTree) resultScroll.scrollTop = previousTree.scrollTop;
    const contextScroll = host.querySelector("[data-context-anchor]");
    if (contextScroll && previousContext?.dataset.contextAnchor === contextScroll.dataset.contextAnchor)
      contextScroll.scrollTop = previousContext.scrollTop;
    const findAction = (root, action, key) => [...root.querySelectorAll("[data-action]")].find(node => node.dataset.action === action && node.dataset.key === key);
    if (focusBack) trigger = findAction(host, focusBack.action, focusBack.key) || host.querySelector("button:not(:disabled)");
    renderDrawer();
    if (view === "search") syncPdfReader();
    const root = insideDrawer && dialog.open ? dialog : host;
    if (action) findAction(root, action, key)?.focus({ preventScroll: true });
    else if (name) {
      const field = [...root.querySelectorAll("[name]")].find(node => node.name === name && (name !== "batchVersion" || node.value === operation));
      field?.focus({ preventScroll: true });
      if (field && cursor) field.setSelectionRange(...cursor);
    } else if (memberCell || memberPick) {
      const cell = [...root.querySelectorAll("[data-member], [data-member-pick]")].find(node =>
        memberCell ? node.dataset.member === memberCell && node.value === operation : node.dataset.memberPick === memberPick);
      cell?.focus({ preventScroll: true });
    }
    if (matrixFrame && state.edit === "members") {
      dialog.querySelector(".member-matrix-frame").scrollLeft = matrixLeft;
      dialog.querySelector(".admin-drawer-body").scrollTop = matrixTop;
    }
    if (state.busy) updateBusy();
    host.setAttribute("aria-busy", String(state.busy));
  }
  async function getRows(version, page = state.page, documentId = null, mode = state.queue ? "reviews" : view) {
    const params = new URLSearchParams({ view: mode, page: String(page) });
    if (version) params.set("version_id", version);
    else if (documentId) params.set("document_id", documentId);
    else {
      if (state.filterTitle.trim()) params.set("title", state.filterTitle.trim());
      const status = mode === "reviews" ? state.queue : state.filterState;
      if (status) params.set("state", status);
      if (state.mine) params.set("mine", "true");
    }
    const data = await request(`${libraryPath()}/documents?${params}`);
    const grouped = data.items.map(item => Object.entries(item.versions).map(([id,value]) =>
      ({ ...value, id, documentId: item.document_id, title: item.title, reviewTarget: allowed(value, "review") ? id : "" })));
    return { rows: mode === "files" && !version && !documentId ? grouped.flatMap(items => {
      const visible = items.find(item => item.status === "PUBLISHED") || items[0];
      return visible ? [{ ...visible, reviewTarget: items.find(item => item.reviewTarget)?.id || "" }] : [];
    }) : grouped.flat(), more: data.has_more };
  }
  async function loadRows(page = state.page) {
    if (!directoryView || !library()) return;
    state.batchIds.clear(); state.batchStrategy = "";
    Object.assign(state, await getRows(null, page), { page });
    if (view === "processing") appliedProgressQuery = progressQuery();
  }
  const progressQuery = () => JSON.stringify([state.libraryId, state.page, state.filterTitle, state.filterState, state.mine]);
  function canRefreshProgress() {
    return !document.hidden && !state.busy && !dialog?.open
      && !document.activeElement?.matches?.(".knowledge-directory-filters :is(input, select, textarea)")
      && library() && appliedProgressQuery === progressQuery()
      && (state.rows.some(row => activeProgress.has(row.progress_stage)) || trackingIndexPanel());
  }
  const trackingIndexPanel = () => state.indexTasks &&
    (state.indexTasks.active_file_rebuild_count > 0 || activeIndexBatch(state.indexTasks.batch));
  async function trackProgress() {
    const row = state.row;
    const trackDetail = Boolean(row && !document.hidden && !state.busy &&
      row.progress_stage?.startsWith("REBUILD") && activeProgress.has(row.progress_stage));
    if (progressLoading || (!trackDetail && !canRefreshProgress())) return;
    const action = directoryAction, query = appliedProgressQuery;
    progressLoading = true;
    try {
      if (trackDetail) {
        const detail = (await getRows(row.id, 1)).rows.find(item => item.id === row.id);
        if (!detail || JSON.stringify(detail) === JSON.stringify(row) || action !== directoryAction || state.row !== row || state.busy) return;
        const rebuild = state.detailIndex ? await request(fileIndexPath(row.id)) : null;
        if (action !== directoryAction || state.row !== row || state.busy) return;
        state.row = detail;
        if (rebuild) state.detailIndex = rebuild;
        state.rows = state.rows.map(item => item.id === row.id
          ? { ...item, progress_stage: detail.progress_stage, index_progress: detail.index_progress } : item);
        render();
        return;
      }
      const data = await getRows();
      if (action !== directoryAction || query !== appliedProgressQuery || !canRefreshProgress()) return;
      if (data.more !== state.more || JSON.stringify(data.rows) !== JSON.stringify(state.rows)) {
        Object.assign(state, data);
        render();
      }
      if (trackingIndexPanel()) {
        const rebuild = await request(libraryPath() + "/index-rebuild");
        if (action === directoryAction && query === appliedProgressQuery && !state.busy && !dialog?.open
          && JSON.stringify(rebuild) !== JSON.stringify(state.indexTasks)) {
          state.indexTasks = rebuild;
          render();
        }
      }
    } catch { /* 暂时失败时保留上次回执，也可手动刷新。 */ }
    finally { progressLoading = false; }
  }
  function load() {
    reset(); state.libraries = []; state.libraryId = ""; render();
    if (management) return refreshCatalog();
    return perform(async epoch => {
      state.libraries = await request("/libraries");
      const requested = linkQuery.get("library_id"), requestedQueue = view === "files" ? linkQuery.get("queue") : "";
      state.libraryId = requested ? eligible().find(item => item.library_id === requested)?.library_id || ""
        : view === "files" ? eligible().find(item => reviewStates(item).includes(requestedQueue))?.library_id || "" : eligible()[0]?.library_id || "";
      if (requested && !state.libraryId) state.error = "指定知识库当前无权访问或已停用，请重新选择。";
      if (reviewStates().includes(requestedQueue)) state.queue = requestedQueue;
      await loadRows();
      const history = new URLSearchParams(location.search);
      if (view === "search" && history.has("history_id")) {
        await openHistory(history.get("history_id"), epoch, history.get("history_hit"));
      }
      if (view === "processing" && linkQuery.get("index_rebuild") === "1" && canIndex("MANAGE")) await refreshIndexTasks();
      const version = linkQuery.get("version_id"), job = linkQuery.get("job_id");
      if (directoryView && library() && version) {
        const result = await getRows(version, 1);
        const row = result.rows.find(row => row.id === version && (!job || job === row.job_id || row.parse_task_history?.some(task => task.job_id === job)));
        if (row) await openRow(row.id);
        else state.error = "指定资料版本或解析任务不可访问。";
      }
    });
  }
  function refreshCatalog() {
    if (state.busy) return;
    state.catalogLoading = true;
    render();
    return perform(async epoch => {
      try { state.libraries = await request("/libraries?management=true"); }
      finally { if (state.epoch === epoch) state.catalogLoading = false; }
    });
  }
  function selectUploads(files) {
    if (state.busy || !state.uploadOpen || !canUpload(library()) || !files?.length) return;
    const rejected = [], formats = library().upload_formats || {};
    for (const file of files) {
      if (state.uploads.length >= 20) { rejected.push("一次清单最多保留 20 个文件。"); break; }
      if (!file.size || file.size > library().upload_max_bytes || file.name.length > 512) {
        rejected.push(file.name + "：空文件、名称过长或超过单文件上限。"); continue;
      }
      const ext = (file.name.split(".").pop() || "").toLowerCase().replace(/^jpeg$/, "jpg").replace(/^tiff$/, "tif");
      const mime = formats[file.type] ? file.type : Object.keys(formats).find(type => formats[type] === ext);
      if (!mime) { rejected.push(file.name + "：不支持此文件格式。"); continue; }
      state.uploads.push({ file, key: api.requestId(), mime });
    }
    state.uploadError = rejected.join(" ");
    render();
  }
  async function upload(epoch) {
    if (!state.uploadOpen || !canUpload(library())) return;
    state.uploadError = "";
    render();
    const path = libraryPath();
    for (const entry of state.uploads.filter(entry => !entry.receipt)) {
      assertCurrent(epoch);
      try {
        entry.receipt = await request(`${path}/documents?name=${enc(entry.file.name)}`, { method: "POST",
          headers: { "Content-Type": entry.mime, "Idempotency-Key": entry.key }, body: entry.file });
        entry.failure = "";
      } catch (error) {
        assertCurrent(epoch);
        entry.failure = error instanceof TypeError || error?.status >= 500
          ? "接收结果未确认，可沿用原提交标识重试。" : error?.message || "提交失败";
      }
    }
    if (state.uploads.some(entry => entry.receipt)) {
      try { await loadRows(1); } catch (error) {
        assertCurrent(epoch);
        state.uploadError = "文件接收结果已保留，但列表刷新失败。可关闭抽屉后刷新。";
      }
    }
  }
  function selectImage(file) {
    revoke(state.queryImageUrl);
    Object.assign(state, { queryImageUrl: "", queryImage: null, error: "", inputKind: "TEXT",
      imageIntent: file ? state.historyEntry?.conditions.image_intent || "SOURCE" : "SOURCE" });
    if (file) {
      if (!/^image\/(png|jpeg|webp)$/.test(file.type) || !file.size || file.size > 8 * 1024 * 1024)
        shell.toast("请选择单张 PNG、JPEG 或 WebP，大小不超过 8 MiB。", "error");
      else { state.queryImage = file; state.queryImageUrl = objectUrl(file); state.inputKind = "IMAGE"; }
    }
    render();
  }
  const searchSeconds = wait => Math.floor((performance.now() - wait.startedAt) / 1000);
  function updateSearchWait(wait) {
    if (searchWait !== wait) return;
    const seconds = searchSeconds(wait);
    const elapsed = host.querySelector("[data-search-elapsed]");
    if (elapsed) elapsed.textContent = `已等待 ${seconds} 秒`;
    const slow = host.querySelector("[data-search-slow]");
    if (slow) slow.hidden = seconds < slowSearchSeconds;
  }
  function endSearchWait(abort = true) {
    const wait = searchWait;
    if (!wait) return;
    searchWait = null;
    clearInterval(wait.timer);
    if (abort) wait.controller.abort();
  }
  function stopSearch() {
    if (!searchWait) return;
    endSearchWait();
    state.notice = "已停止等待。可修改条件或重新检索。";
    render();
    host.querySelector("[data-search-submit]")?.focus({ preventScroll: true });
  }
  function setHistoryLink(id = "", hit = "") {
    const url = new URL(location.href);
    if (id) url.searchParams.set("history_id", id); else url.searchParams.delete("history_id");
    if (id && hit) url.searchParams.set("history_hit", hit); else url.searchParams.delete("history_hit");
    history.replaceState(history.state, "", url);
    state.historyId = id;
  }
  function historyTime(value) {
    const date = new Date(value);
    const time = date.toLocaleTimeString("zh-CN", { hour12: false });
    const pad = number => String(number).padStart(2, "0");
    return date.toDateString() === new Date().toDateString() ? time :
      `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${time}`;
  }
  const imageIntentLabel = value => value === "SIMILAR" ? "找相似图" : "找这张图的出处";
  function historyTitle(conditions) {
    if (conditions.text) return conditions.text;
    if (conditions.image_count) return conditions.image_intent ? imageIntentLabel(conditions.image_intent) : "图片检索";
    const fields = {title:"标题",created_at:"版本创建时间",kind:"类型",document_version_ref:"文档版本",business_asset_ref:"业务资料"};
    const operators = {PREFIX:"开头为",EQ:"等于",NE:"不等于",GE:"不早于",GT:"晚于",LE:"不晚于",LT:"早于"};
    const order = conditions.order_by[0];
    return [...conditions.predicates.map(item => `${fields[item.field] || item.field}${operators[item.operator] || item.operator} ${item.value}`),
      order ? `按${fields[order.field] || "匹配顺序"}${order.descending ? "降序" : "升序"}` : ""].filter(Boolean).join(" · ") || "按条件查找";
  }
  function searchOutputFromTypes(types) {
    const isText = value => value === "TEXT" || value === "TABLE";
    return types.every(isText) ? "TEXT" : types.includes("MIXED") || types.some(isText) ? "ALL" : "IMAGE";
  }
  function searchConditionSummary(output, count, mode) {
    return [output === "TEXT" ? "正文与表格" : output === "IMAGE" ? "只找图片" :
      mode === "RELEVANCE" ? "同时找相关图片" : "结果也包含图片",
    count !== 10 ? `${count} 条主结果` : "", mode === "EXHAUSTIVE" ? "逐页查全" : ""].filter(Boolean).join(" · ");
  }
  async function historyPage(append = false) {
    const data = await request("/search-history?limit=20&offset=" + (append ? state.historyItems.length : 0));
    state.historyItems = append ? [...state.historyItems, ...data.items.filter(item => !state.historyItems.some(old => old.history_id === item.history_id))] : data.items;
    state.historyMore = data.has_more;
  }
  function clearSearchResult() {
    clearPreview(); clearPassage();
    Object.assign(state, { result: null, anchor: "", summary: "", submittedMode: "", submittedImageIntent: null, submittedSearchOptions: "", historyEntry: null, historyStatus: "" });
    state.searchTreeRevision++;
  }
  async function openHistory(id, epoch, hit) {
    clearSearchResult();
    state.historyOpen = false;
    render();
    let detail;
    try { detail = await request("/search-history/" + enc(id)); }
    catch (error) {
      assertCurrent(epoch);
      if ([401, 403, 404, 422].includes(error.status)) {
        setHistoryLink(); state.error = "这条历史不存在、已到期或当前不可访问。"; return;
      }
      throw error;
    }
    const { entry, result, availability } = detail, c = entry.conditions, output = searchOutputFromTypes(c.output_types);
    setHistoryLink(id);
    revoke(state.queryImageUrl);
    const localInput = value => {
      if (!value) return "";
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? "" : new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16);
    };
    const predicate = (field, operator) => c.predicates.find(item => item.field === field && item.operator === operator)?.value;
    Object.assign(state, {
      libraryId: entry.library_name ? entry.library_id : "", historyEntry: entry, result,
      query: c.text || "", mode: c.mode, queryImage: null, queryImageUrl: "", inputKind: "TEXT",
      output,
      count: c.requested_count, prefix: predicate("title", "PREFIX") || "",
      createdFrom: localInput(predicate("created_at", "GE")), createdBefore: localInput(predicate("created_at", "LT")),
      order: c.order_by[0]?.field === "created_at" ? "created_at" : "title", descending: !!c.order_by[0]?.descending,
      searchOptionsOpen: c.mode !== "RELEVANCE",
      summary: [entry.library_name, historyTitle(c)].filter(Boolean).join(" · "),
      submittedImageIntent: c.image_intent || null,
      submittedSearchOptions: searchConditionSummary(output, c.requested_count, c.mode),
      historyStatus: (availability === "UNAVAILABLE" ? "原结果已不可查看。" : availability === "PARTIAL" ? "部分原结果已撤回、换版或不可查看。仅显示仍可访问的首批结果。" : "历史首批结果 · 未重新检索。") +
        (c.image_count > 1 ? " 此记录使用多张图片；此页仅支持单图重新检索。" : c.image_count ? " 重新检索需再选一次原图。" : ""),
    });
    if (hit && evidence(hit)) {
      await readSearchHit(hit, epoch);
      if (state.readerTab === "context") { render(); scrollToSearchMatch(); }
    }
  }
  async function readSearchHit(key, epoch) {
    clearPreview(); clearPassage(); state.anchor = key;
    const item = evidence(key);
    setHistoryLink(state.historyId, key);
    const image = isInlineImage(item);
    if (item?.passage_key || (image && item.locators?.some(loc => loc.page_no > 0))) await loadPassage();
    if (state.passage?.pdf_page) {
      state.readerPage = state.passage.pdf_page; state.readerTab = "pdf"; render();
      await openPassagePdf(epoch);
    } else if (image) await previewEvidence(item, epoch);
  }
  const searchResultPath = ref => `/${state.historyEntry ? "search-history" : "retrievals"}/${enc(ref)}`;
  async function search(saved = null) {
    const selected = library();
    if (!selected || searchWait || state.busy) return;
    const { mode, inputKind, query, prefix, order, descending, output, count, createdFrom, createdBefore, imageIntent } = state;
    const structured = mode !== "RELEVANCE", text = query.trim(), title = prefix.trim();
    const file = (saved ? saved.image_count > 0 : !structured && inputKind === "IMAGE") ? state.queryImage : null;
    const submittedIntent = file ? (saved ? saved.image_intent || "SOURCE" : imageIntent) : null;
    const resultType = saved ? searchOutputFromTypes(saved.output_types) : file ? "IMAGE" : output;
    state.error = ""; state.notice = "";
    let wait;
    try {
      if (saved?.image_count && !file) {
        shell.toast("历史未保留查询图片，请先选择原图。", "error");
        render(); host.querySelector("[data-query-image]")?.focus(); return;
      }
      if (!saved && !structured && !(file || (inputKind === "TEXT" && text))) throw new Error("请填写查询文字或选择一张图片。");
      if (!saved && (!Number.isInteger(count) || count < 1 || count > 100)) {
        state.searchOptionsOpen = true; throw new Error("主结果数必须在 1 至 100 之间。");
      }
      if (!saved && structured && createdFrom && createdBefore && createdFrom >= createdBefore) throw new Error("创建时间上界必须晚于下界。");
      const iso = value => new Date(value).toISOString().replace(/\.000Z$/, "+00:00");
      const predicates = saved?.predicates || (structured ? [
        ...(title ? [{field:"title",operator:"PREFIX",value:title}] : []),
        ...(createdFrom ? [{field:"created_at",operator:"GE",value:iso(createdFrom)}] : []),
        ...(createdBefore ? [{field:"created_at",operator:"LT",value:iso(createdBefore)}] : []),
      ] : []);
      const path = `${libraryPath()}/search`;
      clearSearchResult(); setHistoryLink();
      state.submittedSearchOptions = [submittedIntent ? imageIntentLabel(submittedIntent) : "",
        searchConditionSummary(resultType, saved?.requested_count ?? count, saved?.mode || mode)].filter(Boolean).join(" · ");
      Object.assign(state, { result: null, anchor: "", submittedMode: saved?.mode || mode, submittedImageIntent: submittedIntent,
        summary: `${selected.display_name} · ${saved ? historyTitle(saved) : structured ? `${title || "全部标题"} · ${order === "title" ? "标题" : "版本创建时间"}${descending ? "降序" : "升序"}${createdFrom || createdBefore ? ` · ${createdFrom || "不限"} 至 ${createdBefore || "不限"}（本地时间，上界不含）` : ""}` : file?.name || text}` });
      wait = searchWait = { controller: new AbortController(), startedAt: performance.now(), timer: 0 };
      wait.timer = setInterval(() => updateSearchWait(wait), 1000);
      render();
      const images = [];
      if (file) {
        const raw = new Uint8Array(await file.arrayBuffer()); let binary = "";
        if (searchWait !== wait) return;
        for (let start = 0; start < raw.length; start += 32768) binary += String.fromCharCode(...raw.subarray(start, start + 32768));
        images.push({ mime_type: file.type, image_base64: btoa(binary) });
      }
      const result = await request(path, { method: "POST", headers: { "Content-Type": "application/json" },
        signal: wait.controller.signal, body: JSON.stringify({
          mode: saved?.mode || mode, text: saved ? saved.text : structured || file ? null : text,
          images: saved && !saved.image_count ? [] : images, requested_count: saved?.requested_count ?? count,
          ...(submittedIntent ? { image_intent: submittedIntent } : {}),
          output_types: saved?.output_types || (resultType === "TEXT" ? ["TEXT","TABLE"] : resultType === "IMAGE" ? ["FIGURE","PHOTO","PAGE","REGION"] : ["TEXT","TABLE","FIGURE","PHOTO","PAGE"]),
          predicates, order_by: saved?.order_by || (structured ? [{field:order,descending}] : []), combination: saved?.combination ?? null,
        }) });
      if (searchWait !== wait) return;
      if (result.status === "FAILED") {
        const timedOut = result.reason === "TIMEOUT" || result.channels?.some(channel => channel.status === "FAILED" && channel.reason === "TIMEOUT");
        shell.toast(timedOut ? "有检索通道超时，请稍后重试。" : "检索通道执行失败，请稍后重试。", "error");
        return;
      }
      state.result = result;
      setHistoryLink(result.history_id || "");
    } catch (error) {
      if ((wait && searchWait !== wait) || error?.name === "AbortError") return;
      const message = error?.code === "MODALITY_UNAVAILABLE" && submittedIntent === "SIMILAR" ? "当前知识库没有已发布的图片相似索引，可切换到「找这张图的出处」。" :
        error?.code === "TIMEOUT" ? "检索超时，请稍后重试。" :
        ["SCOPE_CHANGED","SNAPSHOT_STALE","CURSOR_EXPIRED","ARTIFACT_EXPIRED"].includes(error?.code)
          ? "检索回执已失效，请重新检索。" : error?.message || "检索未完成，请重试。";
      shell.toast(message, "error");
    } finally {
      if (!wait || searchWait === wait) {
        const stopFocused = document.activeElement?.dataset.action === "stop-search";
        endSearchWait(false);
        render();
        if (stopFocused) host.querySelector("[data-search-submit]")?.focus({ preventScroll: true });
      }
    }
  }
  async function nextResults(key) {
    const current = state.result, related = key ? group(key) : null, cursor = related ? related.continuation : current?.continuation;
    if (state.historyEntry || !current || !cursor) return;
    const next = await json(`/retrievals/${enc(related?.retrieval_ref || current.retrieval_ref)}${related ? "/relations:continue" : ":continue"}`,
      "POST", related ? { cursor, anchor_key: key } : { cursor });
    if (!related) { state.result = next; state.searchTreeRevision++; state.anchor = ""; setHistoryLink(state.historyId); clearPreview(); clearPassage(); return; }
    const nextGroup = next.linked.find(value => value.anchor_key === key);
    if (!nextGroup) throw new Error("未返回对应关联组，请重新检索。");
    const items = new Map(current.items.map(item => [item.key, item]));
    next.items.forEach(item => { if (!items.has(item.key)) items.set(item.key, item); });
    state.result = { ...current, items: [...items.values()], linked: current.linked.map(value => value.anchor_key === key ?
      { ...nextGroup, retrieval_ref: next.retrieval_ref, linked_keys: [...new Set([...value.linked_keys, ...nextGroup.linked_keys])] } : value) };
  }
  async function previewEvidence(entry, epoch) {
    if (!canPreview(entry)) return;
    const { retrieval_ref, key } = entry.content_ref;
    const result = state.result, anchor = state.anchor;
    const blob = await api.requestBlob(`/knowledge${searchResultPath(retrieval_ref)}/content/${enc(key)}`,
      { signal: state.controller.signal, headers: { Accept: entry.mime_type } });
    const matchesSelection = () => state.result === result && state.anchor === anchor && evidence(entry.key) === entry;
    const mime = blob.type.split(";")[0];
    if (mime !== entry.mime_type || blob.size !== entry.byte_count) throw new Error("证据预览格式不匹配。");
    if (isInlineImage(entry)) {
      const url = objectUrl(blob);
      try {
        const image = new Image(); image.src = url;
        await image.decode().catch(() => { throw new Error("图片无法在浏览器中预览。"); });
        assertCurrent(epoch);
        if (!matchesSelection()) { revoke(url); return; }
        clearPreview(); state.preview = url; state.previewKey = entry.key;
      } catch (error) { revoke(url); throw error; }
    } else {
      const value = await blob.text();
      assertCurrent(epoch);
      if (matchesSelection()) entry.text = value;
    }
  }
  function passagePath(anchor) {
    const ref = anchor?.content_ref;
    if (!ref) throw new Error("当前结果没有可读取的段落定位。");
    return `${searchResultPath(ref.retrieval_ref)}/context/${enc(ref.key)}`;
  }
  async function loadPassage(before = state.passageBefore, after = state.passageAfter) {
    const data = await request(passagePath(evidence(state.anchor)) + "?" + new URLSearchParams({ before, after }));
    Object.assign(state, { passage: data, passageBefore: before, passageAfter: after });
  }
  async function openPassagePdf(epoch) {
    const blob = await api.requestBlob("/knowledge" + passagePath(evidence(state.anchor)) + "/pdf",
      { signal: state.controller.signal, headers: { Accept: "application/pdf" } });
    assertCurrent(epoch);
    if (blob.type.split(";")[0] !== "application/pdf" || await blob.slice(0, 5).text() !== "%PDF-")
      throw new Error("PDF 原文不可预览。");
    assertCurrent(epoch);
    revoke(state.passagePdf); state.passagePdf = objectUrl(blob);
  }
  const filePreviewType = /^(?:image\/(?:png|jpeg|webp)|application\/pdf|text\/(?:plain|markdown))$/;
  async function loadFilePreview(row) {
    clearFilePreview();
    if (!filePreviewType.test(row.mime_type) || (row.mime_type === "application/pdf" && navigator.pdfViewerEnabled === false)) {
      state.filePreviewState = "unavailable"; return;
    }
    const controller = new AbortController();
    state.filePreviewController = controller; state.filePreviewState = "loading";
    try {
      const blob = await api.requestBlob(`/knowledge${libraryPath()}/versions/${enc(row.id)}/source?preview=true`,
        { signal: controller.signal, headers: { Accept: row.mime_type } });
      const mime = blob.type.split(";")[0].toLowerCase();
      if (mime !== row.mime_type || !filePreviewType.test(mime)) throw new Error("预览格式不匹配");
      let data;
      if (mime.startsWith("text/")) data = { kind: "text", content: await blob.slice(0, 65536).text(), truncated: blob.size > 65536 };
      else {
        if (mime === "application/pdf" && await blob.slice(0, 5).text() !== "%PDF-") throw new Error("无效 PDF");
        if (state.filePreviewController !== controller) return;
        data = { kind: mime === "application/pdf" ? "pdf" : "image", content: objectUrl(blob.slice(0, blob.size, mime)) };
      }
      if (state.filePreviewController === controller) { state.filePreview = data; state.filePreviewState = ""; }
    } catch (error) {
      if (state.filePreviewController === controller) state.filePreviewState = error.code === "KNOWLEDGE_PREVIEW_UNAVAILABLE" ? "unavailable" : "error";
    } finally {
      if (state.filePreviewController === controller && !state.busy) render();
    }
  }
  function filePreviewHtml(row) {
    const current = state.filePreview;
    const content = current?.kind === "image" ? `<img class="knowledge-preview" data-file-preview-image src="${esc(current.content)}" alt="${esc(row.title)}" draggable="false">`
      : current?.kind === "pdf" ? `<iframe src="${esc(current.content)}#toolbar=0" title="当前版本 PDF 原文预览" tabindex="0" referrerpolicy="no-referrer"></iframe>`
      : current?.kind === "text" ? `<pre>${esc(current.content)}</pre>${current.truncated ? "<small>仅预览前 64 KiB。</small>" : ""}`
      : state.filePreviewState === "loading" ? '<p role="status">正在加载此版本的预览…</p>'
      : state.filePreviewState === "unavailable" ? '<p>此文件格式或当前浏览器不支持内联预览。</p>'
      : state.filePreviewState === "error" ? '<p role="status">原文预览未能加载。</p>' + button("retry-file-preview", "重试预览", "", "row-action") : "";
    return `<section class="drawer-form-section knowledge-file-preview" aria-label="原文预览"><h3>原文预览</h3>${content}</section>`;
  }
  async function versionAction(action, epoch) {
    const row = state.row;
    const operation = ["approve","reject"].includes(action) ? "review" : action;
    if (!row || !allowed(row, operation) || (action === "approve" && !canApprove(row)) ||
        (operation === "reprocess" && !canReprocess(row))) return;
    const names = { cancel: "取消任务", withdraw: "撤回版本", reprocess: "重新解析", "retry-index": "重试建索引" };
    const maintenance = ["reprocess", "retry-index"].includes(operation);
    if (operation !== "review" && !await shell.confirmAction({
      title: names[operation] + "？", confirmLabel: names[operation], tone: maintenance ? "warning" : "danger",
      message: operation === "withdraw" ? "撤回后停止检索，历史引用不会替换到新版。保留文件。" :
        operation === "reprocess" ? (row.status === "PUBLISHED" ? "将重新读取原文件并建索引。新索引校验通过后自动切换，期间旧版仍可检索。" :
          "将创建新解析任务，保留原任务和原文件。索引校验通过后自动发布。") :
        operation === "retry-index" ? "使用已有解析结果重试建索引，不重新解析。索引校验通过后自动发布。" : "取消后不会继续接收该任务的解析结果。",
    })) return;
    assertCurrent(epoch);
    await json(`${libraryPath()}/versions/${enc(row.id)}:${operation}`, "POST",
      operation === "review" ? { decision: action === "approve" ? "APPROVE" : "REJECT", comment: state.comment.trim() || null,
        parse_strategy: action === "approve" && row.parse_status === "AWAITING_APPROVAL" ? state.parseStrategy : undefined }
        : operation === "reprocess" ? { expected_job_id: row.job_id, mode: state.reprocessMode,
            parse_strategy: state.reprocessMode === "CURRENT" ? state.parseStrategy : undefined }
        : operation === "retry-index" ? { expected_job_id: row.job_id } : undefined);
    state.row = null; clearPreview(); clearFilePreview(); await loadRows();
    state.notice = operation === "review" ? "审核决定已保存。" : maintenance ? `${names[operation]}任务已排队。` : `已${names[operation]}。`;
  }
  async function openRow(id, governing = false) {
    if (governing && !reviewStates().length) return;
    const mode = governing || state.queue ? "reviews" : view;
    const details = await getRows(id, 1, null, mode);
    const exact = details.rows.find(item => item.id === id);
    if (!exact) throw new Error("此版本已不可访问，请刷新列表。");
    state.fileVersions = view === "files" && !governing ? (await getRows(null, 1, exact.documentId, mode)).rows : [];
    state.governing = governing;
    state.row = exact; state.comment = "";
    state.detailIndex = null;
    state.reprocessMode = exact.original_parse_available ? "ORIGINAL" : "CURRENT";
    state.parseStrategy = ((governing && allowed(exact, "review") && awaitingApproval(exact)) || allowed(exact, "reprocess")) &&
      exact.parse_strategy_options?.some(option => option.strategy === "AUTO" && option.selectable) ? "AUTO" : "";
    clearPreview();
    if (view === "processing" && canIndex("MANAGE")) await refreshFileIndex();
    if (view === "files") void loadFilePreview(exact);
  }
  async function batchReview(decision, epoch) {
    const rows = selectedReviews(), strategy = state.batchStrategy;
    if (state.queue !== "PENDING" || !rows.length ||
      (decision === "APPROVE" && rows.some(awaitingApproval) && !commonStrategies(rows).includes(strategy))) return;
    const action = decision === "APPROVE" ? "批量通过" : "批量拒绝";
    if (!await shell.confirmAction({
      title: `${action} ${rows.length} 个版本？`,
      message: decision === "APPROVE" ? `${rows.some(awaitingApproval) ? `解析方案：${strategyLabels[strategy]}。 ` : ""}逐个提交所选版本。首次处理的资料审核通过后开始解析，索引校验通过后自动发布。` : "逐个拒绝所选版本。等待审批的版本不会进入解析。",
      confirmLabel: action, tone: "warning",
    })) return;
    const failures = [];
    for (const row of rows) {
      assertCurrent(epoch);
      try {
        await json(`${libraryPath()}/versions/${enc(row.id)}:review`, "POST", { decision,
          parse_strategy: decision === "APPROVE" && awaitingApproval(row) ? strategy : undefined });
        state.batchIds.delete(row.id);
      } catch {
        assertCurrent(epoch);
        failures.push(`${row.title} · ${row.id}：未确认成功，请刷新后核对状态。`);
      }
    }
    state.batchFailures = failures;
    state.notice = `${action}：${rows.length - failures.length} 个成功，${failures.length} 个未确认。`;
    await loadRows();
  }
  async function toggle(item, epoch) {
    const disabling = item.status === "ACTIVE", text = disabling ? "停用" : "启用";
    if (!await shell.confirmAction({ title: `${text}知识库？`, confirmLabel: text, tone: disabling ? "danger" : "warning",
      message: disabling ? `停用「${item.display_name}」后，不能上传、检索或新绑定，原资料保留。` : `按现有授权恢复「${item.display_name}」的使用。` })) return;
    assertCurrent(epoch);
    const next = await json("/libraries/" + enc(item.library_id), "PATCH", { status: disabling ? "DISABLED" : "ACTIVE", expected_revision: item.row_version });
    state.libraries = state.libraries.map(value => value.library_id === item.library_id ? next : value);
    state.notice = `知识库已${text}。`;
  }
  function modelFields(roles) {
    return roles.map(([role, category, label]) => {
      const models = (state.modelCatalog || []).filter(item => Number(item.category) === category);
      const bound = state[role];
      const entries = [["", "不配置"], ...(bound && !models.some(item => item.model_id === bound) ? [[bound, "保留当前绑定（不在可选目录）"]] : []),
        ...models.map(item => {
          const direction = role === "visual_embedding" ? globalThis.KBotKnowledgeCapabilitiesReview.textToImageLabel(item) : "";
          return [item.model_id, [item.display_name, item.provider, direction].filter(Boolean).join(" · ")];
        })];
      return field(label + " 模型（可选）", select(role, entries, bound));
    }).join("");
  }

  async function save(epoch, choice = "") {
    if (!state.name.trim()) throw new Error("请填写知识库名称。");
    if (state.edit === "create" && !state.confirmingCreate) {
      state.confirmingCreate = true;
      return;
    }
    const item = state.managed, changes = { display_name: state.name.trim(), description: state.description.trim() || null };
    if (item && changedIndexes().length && !choice) {
      const current = await request(`/libraries/${enc(item.library_id)}?management=true`);
      if (current.row_version !== item.row_version || !current.index_maintenance) throw new Error("请刷新知识库后重新确认变更。");
      state.indexImpact = current.index_maintenance;
      return;
    }
    for (const [roles, group] of [[parseRoles, "parse_models"], [indexRoles, "index_models"]]) {
      if (modelsChanged(roles, group)) changes[group] = Object.fromEntries(roles.map(([role]) => [role, state[role] || null]));
    }
    if (choice === "rebuild") changes.rebuild_indexes = true;
    assertCurrent(epoch);
    const next = item ? await json("/libraries/" + enc(item.library_id), "PATCH", { ...changes, expected_revision: item.row_version })
      : await json("/libraries", "POST", changes);
    state.libraries = [next, ...state.libraries.filter(value => next.library_id !== value.library_id)];
    state.edit = ""; state.indexImpact = null;
    state.notice = choice === "rebuild" ? "知识库已保存，索引重建已排队。" : item ? "知识库已保存。" : "知识库已创建。";
  }
  async function refreshIndexTasks() {
    if (canIndex("MANAGE")) state.indexTasks = await request(libraryPath() + "/index-rebuild");
  }
  const fileIndexPath = id => libraryPath() + "/index-rebuild?version_id=" + enc(id);
  async function refreshFileIndex() {
    const row = state.row;
    if (row && canIndex("MANAGE")) state.detailIndex = await request(fileIndexPath(row.id));
  }
  async function indexAction(action, key, epoch, file = false) {
    const data = file ? state.detailIndex : state.indexTasks, batch = data?.batch, row = state.row;
    if (!data || (action !== "START" && !batch) || !canIndex("MANAGE")) return;
    if (file && action === "START" && (!row?.generation_id || !allowed(row, "rebuild-index"))) return;
    const titles = { START: file ? "重建索引" : "开始重建", RETRY: file ? "重试重建索引" : "重试此文件", DISCARD: "放弃批次" };
    const messages = { START: `为 ${data.published_file_count} 个文件重建索引；全部校验通过后自动切换，期间旧索引继续提供检索。`,
      RETRY: "仅重新处理该失败文件，不影响旧索引。",
      DISCARD: "放弃本批次的候选索引，当前已发布索引不变。" };
    if (!(file && action === "START") &&
      !await shell.confirmAction({ title: titles[action] + "？", message: messages[action], confirmLabel: titles[action], tone: "warning" })) return;
    assertCurrent(epoch);
    const result = await json(libraryPath() + "/index-rebuild", "POST", action === "START"
      ? (file ? { action, version_id: row.id, expected_generation_id: row.generation_id } : { action, expected_revision: data.library_revision })
      : { action, batch_id: batch.batch_id, version_id: key || undefined });
    if (file) {
      if (state.indexTasks) state.indexTasks = { ...state.indexTasks, active_file_rebuild_count: result.active_file_rebuild_count };
      if (action === "START") {
        const queued = result.batch?.items.find(item => item.version_id === row.id);
        if (queued) state.rows = state.rows.map(item => item.id === row.id
          ? { ...item, progress_stage: "REBUILD_QUEUED", index_progress: queued.index_progress } : item);
        await closeDrawer(epoch);
        state.detailIndex = null;
        state.notice = "索引重建已排队，可在表格查看进度。";
        render();
      } else {
        state.detailIndex = result;
      }
    } else {
      state.indexTasks = result;
    }
    await loadRows();
    if (file && action !== "START") state.row = (await getRows(row.id, 1)).rows.find(item => item.id === row.id) || row;
  }
  async function closeDrawer(epoch) {
    if (state.historyOpen) { state.historyOpen = false; return; }
    if (state.readerExpanded) { state.readerExpanded = false; return; }
    const item = state.managed;
    if (state.uploadOpen && state.uploads.some(entry => !entry.receipt) &&
      !await shell.confirmAction({ title: "放弃未完成的文件提交？", message: "关闭会清空待提交清单，已收到的文件不会撤回。", confirmLabel: "放弃并关闭" })) return;
    const dirty = (state.edit === "members" && membersDirty()) ||
      (["create","edit"].includes(state.edit) && (state.name !== (item?.display_name || "") || state.description !== (item?.description || "") || modelsChanged(parseRoles, "parse_models") || modelsChanged(indexRoles, "index_models")));
    if (dirty &&
      !await shell.confirmAction({ title: "放弃未保存的修改？", message: "关闭后本次输入不会保留。", confirmLabel: "放弃修改", tone: "warning" })) return;
    assertCurrent(epoch);
    state.anchor = ""; state.row = null; state.edit = ""; state.progressHelp = false; clearPreview(); clearFilePreview(); clearPassage();
    state.uploadOpen = false; state.uploads = []; state.uploadError = "";
  }
  async function loadMemberMatrix(epoch) {
    const path = "/libraries/" + enc(state.managed.library_id);
    render();
    const [snapshot, candidates] = await Promise.allSettled([request(path + "?management=true"), request(path + "/member-options")]);
    assertCurrent(epoch);
    if (snapshot.status === "rejected") throw snapshot.reason;
    const fresh = snapshot.value;
    if (!state.memberMatrix.blocked) state.memberMatrix = matrix.createMemberMatrix(fresh.grants);
    if (candidates.status === "rejected") throw candidates.reason;
    if (state.memberMatrix.blocked) matrix.refreshMatrix(state.memberMatrix, fresh.grants, candidates.value);
    else matrix.setOptions(state.memberMatrix, candidates.value);
    state.managed = fresh;
    state.libraries = state.libraries.map(item => item.library_id === fresh.library_id ? fresh : item);
  }
  async function runAction(action, key, epoch) {
    switch (action) {
      case "close": return closeDrawer(epoch);
      case "history-more": return historyPage(true);
      case "history-view": return openHistory(key, epoch);
      case "history-delete":
        if (!await shell.confirmAction({ title: "删除这条检索历史？", message: "只移除这条记录，不影响知识库资料。", confirmLabel: "删除", tone: "danger" })) return;
        assertCurrent(epoch);
        await json("/search-history/" + enc(key), "DELETE");
        if (state.historyId === key) { clearSearchResult(); setHistoryLink(); }
        return historyPage();
      case "index-refresh": return refreshIndexTasks();
      case "index-start": return indexAction("START", key, epoch);
      case "index-retry": return indexAction("RETRY", key, epoch);
      case "index-discard": return indexAction("DISCARD", key, epoch);
      case "file-index-refresh": return refreshFileIndex();
      case "file-index-start": return indexAction("START", key, epoch, true);
      case "file-index-retry": return indexAction("RETRY", key, epoch, true);
      case "file-index-discard": return indexAction("DISCARD", key, epoch, true);
      case "index-back": state.indexImpact = null; return;
      case "index-save": return save(epoch, "save");
      case "index-save-rebuild": return save(epoch, "rebuild");
      case "reload-members": return loadMemberMatrix(epoch);
      case "remove-member": {
        const m = state.memberMatrix;
        if (!m?.options || m.blocked || matrix.protectsManager(m, key)) return;
        if (!await shell.confirmAction({ title: "移除这位成员？",
          message: `「${matrix.personName(m, key)}（${key}）」将失去本库授权。只在最终保存后生效。`,
          confirmLabel: "暂存移除", tone: "danger" })) return;
        assertCurrent(epoch);
        matrix.remove(m, key);
        return;
      }
      case "back-create": state.confirmingCreate = false; return;
      case "next": return loadRows(state.page + 1);
      case "previous": return loadRows(state.page - 1);
      case "refresh-progress": return loadRows();
      case "queue":
        if (state.queue === key || (key && !reviewStates().includes(key))) return;
        Object.assign(state, { queue: key, batchFailures: [], rows: [], more: false, page: 1 });
        return loadRows(1);
      case "batch-approve": case "batch-reject": return batchReview(action === "batch-approve" ? "APPROVE" : "REJECT", epoch);
      case "continue": return nextResults();
      case "relations": return nextResults(key);
      case "context-retry": return loadPassage();
      case "context-before": return loadPassage(Math.min(10, state.passageBefore + 2), state.passageAfter);
      case "context-after": return loadPassage(state.passageBefore, Math.min(10, state.passageAfter + 2));
      case "context-pdf": return openPassagePdf(epoch);
      case "content": return previewEvidence(evidence(key), epoch);
      case "approve": case "reject": case "cancel": case "withdraw":
      case "reprocess": case "retry-index": return versionAction(action, epoch);
      case "toggle": {
        const item = eligible().find(value => value.library_id === key);
        if (item) return toggle(item, epoch);
        return;
      }
      case "delete": {
        const item = management && eligible().find(value => value.library_id === key);
        if (!item) return;
        const path = "/libraries/" + enc(key);
        const current = await request(`${path}?management=true`);
        if (current.row_version !== item.row_version) throw new Error("知识库已变化，请刷新后重试。");
        if (current.delete_block !== null) throw new Error(deleteReasons[current.delete_block] || "暂时无法确认知识库能否删除，请刷新后重试。");
        if (!await shell.confirmAction({ title: "删除知识库？", message: `「${item.display_name}」将永久移除。提交时将再次检查资料及历史引用。`, confirmLabel: "删除知识库", tone: "danger" })) return;
        assertCurrent(epoch);
        await request(path, { method: "DELETE", headers: { "If-Match": `"rv-${item.row_version}"` } });
        state.libraries = state.libraries.filter(entry => entry.library_id !== key);
        state.notice = "知识库已删除。";
        return;
      }
      case "save-members": {
        const item = state.managed;
        if (!item || !memberReady() || !membersDirty()) return;
        const m = state.memberMatrix;
        if (!await shell.confirmAction(matrix.reviewMembers(m))) return;
        assertCurrent(epoch);
        let next;
        try {
          next = await json("/libraries/" + enc(item.library_id), "PATCH", { grants: m.draft, expected_revision: item.row_version });
        } catch (error) {
          if (["KNOWLEDGE_LIBRARY_REVISION_CONFLICT", "KNOWLEDGE_MEMBER_INELIGIBLE"].includes(error.code)) {
            m.blocked = true;
            m.note = "";
            throw new Error("授权版本或成员资格已变化，本次改动仍在草稿中。请读取最新授权后重新复核。");
          }
          throw error;
        }
        state.libraries = can(next,"MANAGE") ? state.libraries.map(value => value.library_id === item.library_id ? next : value)
          : state.libraries.filter(value => value.library_id !== item.library_id);
        state.edit = ""; state.notice = "成员授权已更新。";
      }
    }
  }
  function bind(root) {
    root.addEventListener("input", event => {
      const { name, value, checked } = event.target;
      if (name === "memberQuery" && !event.isComposing && !state.busy) {
        state.memberMatrix.query = value; state.memberMatrix.page = 1; render(); return;
      }
      if (["query","prefix","order","name","description","filterTitle","filterState","comment","createdFrom","createdBefore"].includes(name)) state[name] = value;
      if (management && root === host && name === "filterTitle" && !event.isComposing) filterCatalog();
      if (name === "count") state.count = Number(value);
      if (["descending","mine"].includes(name)) state[name] = checked;
    });
    root.addEventListener("compositionend", event => {
      if (event.target.name !== "memberQuery" || state.busy) return;
      state.memberMatrix.query = event.target.value; state.memberMatrix.page = 1; render();
    });
    root.addEventListener("change", event => {
      const field = event.target;
      if (field.name === "batchVersion" && !state.busy) {
        if (field.checked) state.batchIds.add(field.value); else state.batchIds.delete(field.value);
        if (!commonStrategies(selectedReviews()).includes(state.batchStrategy)) state.batchStrategy = "";
        render();
      } else if (field.name === "batchStrategy" && !state.busy) {
        state.batchStrategy = field.value; render();
      } else if (field.matches("[data-library]")) {
        reset(); state.libraryId = field.value;
        if (view === "search") setHistoryLink();
        render(); void perform(() => loadRows(1));
      } else if (management && root === host && field.name === "filterState") {
        state.filterState = field.value; filterCatalog();
      } else if (field.matches("[data-upload]")) selectUploads(field.files);
      else if (field.matches("[data-query-image]")) selectImage(field.files[0]);
      else if (field.matches("[data-member], [data-member-pick], [name=memberBatchOp]") && !state.busy && !state.memberMatrix.blocked) {
        const m = state.memberMatrix;
        if (field.name === "memberBatchOp") { m.batchOp = field.value; return; }
        if (field.dataset.member) matrix.adjust(m, field.dataset.member, field.value, field.checked);
        else {
          const selection = matrix.selected(m);
          if (field.checked) selection.add(field.dataset.memberPick);
          else selection.delete(field.dataset.memberPick);
        }
        render();
      }
      else if (field.name === "parseStrategy") {
        state.parseStrategy = field.value;
        const approval = dialog.querySelector('[data-action="approve"]');
        const retry = dialog.querySelector('[data-action="reprocess"]');
        if (approval) approval.disabled = !canApprove(state.row);
        if (retry) retry.disabled = !canReprocess(state.row);
      }
      else if (field.name === "reprocessMode") { state.reprocessMode = field.value; render(); }
      else if ([...parseRoles, ...indexRoles].some(([role]) => field.name === role)) state[field.name] = field.value;
      else if (field.name === "fileVersion") {
        const row = state.fileVersions.find(value => value.id === field.value);
        if (row) { state.row = row; clearPreview(); void loadFilePreview(row); render(); }
      }
      else if (field.name === "imageIntent") { state.imageIntent = field.value; }
      else if (field.name === "includeImages" || field.name === "onlyImages") {
        state.output = !field.checked ? "TEXT" : field.name === "includeImages" ? "ALL" : "IMAGE";
        render();
      } else if (field.name === "structuredSearch") {
        state.mode = field.checked ? "FILTERED_ORDER" : "RELEVANCE";
        if (field.checked) selectImage(); else render();
      } else if (field.name === "exhaustiveSearch") {
        state.mode = field.checked ? "EXHAUSTIVE" : "FILTERED_ORDER"; render();
      }
    });
    root.addEventListener("submit", event => {
      event.preventDefault();
      const form = event.target.dataset.form;
      if (form === "pdf-page") {
        const page = event.target.elements.pdfPage.valueAsNumber;
        if (Number.isSafeInteger(page) && page > 0 && (!pdfReader?.pageCount || page <= pdfReader.pageCount)) {
          state.readerPage = page; render();
        }
        return;
      }
      if (form === "search") return void search();
      const task = { upload, filter: () => loadRows(1), save }[form];
      if (task) void perform(task);
    });
    root.addEventListener("click", event => {
      const node = event.target.closest("[data-action]");
      if (!node || state.busy) return;
      const { action, key } = node.dataset;
      if (action === "stop-search") return stopSearch();
      if (action === "search-history") {
        if (searchWait) return;
        trigger = node; state.historyOpen = true; state.historyItems = []; state.historyMore = false;
        void perform(async () => { render(); await historyPage(); });
        return;
      }
      if (action === "history-rerun") return void search(state.historyEntry?.conditions);
      if (action === "pick-library") {
        if (key === state.libraryId || !eligible().some(item => item.library_id === key)) return;
        reset(); state.libraryId = key;
        render();
        return void perform(() => loadRows(1))?.then(() => {
          if (state.libraryId === key && document.activeElement === document.body)
            [...host.querySelectorAll('[data-action="pick-library"]')].find(item => item.dataset.key === key)?.focus();
        });
      }
      if (action === "add-file") {
        if (canUpload(library())) { trigger = node; state.uploadOpen = true; render(); }
        return;
      }
      if (action === "progress-help") { trigger = node; state.progressHelp = true; renderDrawer(); return; }
      if (action === "retry-file-preview") {
        if (state.row) { void loadFilePreview(state.row); render(); }
        return;
      }
      if (action === "reload") return management ? refreshCatalog() : load();
      if (action === "catalog-more") {
        const opening = node.getAttribute("aria-expanded") !== "true";
        closeCatalogMenus();
        if (opening) { node.setAttribute("aria-expanded", "true"); node.nextElementSibling.hidden = false; }
        return;
      }
      if (action === "remove-image") return selectImage();
      if (action === "search-options") { state.searchOptionsOpen = !state.searchOptionsOpen; return render(); }
      if (action === "reader-tab") {
        state.readerTab = key; render();
        if (key === "context") scrollToSearchMatch();
        return;
      }
      if (action === "reader-expand") { trigger = node; state.readerExpanded = true; return render(); }
      if (action === "pdf-hit") { state.readerPage = state.passage.pdf_page; return render(); }
      if (action === "pdf-step") {
        const page = state.readerPage + Number(key);
        if (page > 0 && page <= pdfReader?.pageCount) { state.readerPage = page; render(); }
        return;
      }
      if (action === "pdf-fit") { state.readerZoom = 0; return render(); }
      if (action === "pdf-zoom") {
        state.readerZoom = Math.max(50, Math.min(300, (state.readerZoom || 100) + Number(key))); return render();
      }
      if (action === "pick-upload") { root.querySelector("[data-upload]")?.click(); return; }
      if (action === "remove-upload") { state.uploads = state.uploads.filter(entry => entry.key !== key); return render(); }
      if (["member-tab", "member-prev", "member-next", "member-clear", "member-batch-add", "member-grant", "member-revoke", "add-member"].includes(action)) {
        const m = state.memberMatrix;
        if (action === "member-tab") { m.tab = key; m.page = 1; }
        if (action === "member-prev" || action === "member-next") m.page = matrix.matrixPage(m).page + (action === "member-prev" ? -1 : 1);
        if (action === "member-clear") matrix.selected(m).clear();
        if (action === "member-batch-add" || action === "add-member") matrix.join(m, action === "add-member" ? [key] : matrix.selected(m));
        if (action === "member-grant" || action === "member-revoke") matrix.batch(m, action === "member-grant");
        render();
        if (!dialog.contains(document.activeElement) || document.activeElement.matches('.admin-drawer-header [data-action="close"]'))
          dialog.querySelector('[name="memberQuery"]')?.focus({ preventScroll: true });
        return;
      }
      if (action === "delete" || action === "toggle") {
        closeCatalogMenus(true);
        const epoch = state.epoch;
        void perform(generation => runAction(action, key, generation)).then(() => {
          if (epoch !== state.epoch || document.activeElement !== document.body) return;
          const opener = [...host.querySelectorAll(".management-catalog-more")].find(button => button.dataset.key === key);
          (opener || host.querySelector(".management-catalog-add-card"))?.focus({ preventScroll: true });
        });
        return;
      }
      if (["evidence", "row", "govern", "create", "edit", "members"].includes(action)) {
        if (action === "evidence" && state.anchor === key) return;
        trigger = node.closest(".management-catalog-menu")?.previousElementSibling || node;
        if (trigger !== node) { trigger.focus(); closeCatalogMenus(); }
        state.error = ""; clearPreview();
        if (action === "evidence") {
          if (evidence(key)) {
            void perform(async epoch => {
              const loading = readSearchHit(key, epoch);
              render(); await loading;
            })?.then(() => {
              if (state.anchor !== key) return;
              if (state.readerTab === "context") scrollToSearchMatch();
              if (matchMedia("(max-width: 860px)").matches) host.querySelector(".knowledge-search-reader")?.focus();
              else if (document.activeElement === document.body)
                [...host.querySelectorAll('[data-action="evidence"]')].find(item => item.dataset.key === key)?.focus({ preventScroll: true });
            });
            return;
          }
        }
        else if (action === "row" || action === "govern") { void perform(() => openRow(key, action === "govern")); return; }
        else {
          const item = eligible().find(value => value.library_id === key);
          if (!item && action !== "create") return;
          Object.assign(state, { edit: action, confirmingCreate: false, indexImpact: null, managed: item, name: item?.display_name || "", description: item?.description || "",
            ocr: item?.parse_models?.ocr || "", vlm: item?.parse_models?.vlm || "", modelCatalog: [],
            embedding: item?.index_models?.embedding || "", visual_embedding: item?.index_models?.visual_embedding || "",
            memberMatrix: action === "members" ? matrix.createMemberMatrix(item.grants) : null });
        }
        render();
        if (action === "evidence" && matchMedia("(max-width: 860px)").matches)
          host.querySelector(".knowledge-search-reader")?.focus();
        if (action === "create" || action === "edit") void perform(async epoch => {
          const models = await api.request("/model-catalog", { signal: state.controller.signal });
          assertCurrent(epoch); state.modelCatalog = models;
        });
        if (action === "members") void perform(loadMemberMatrix);
        return;
      }
      const current = state.epoch, fromMembers = state.edit === "members";
      void perform(epoch => runAction(action, key, epoch))?.then(() => {
        if (!fromMembers || current !== state.epoch || state.edit !== "members") return;
        const target = [...dialog.querySelectorAll("[data-action]:not(:disabled)")].find(control =>
          state.memberMatrix.blocked && action === "save-members" ? control.dataset.action === "reload-members" :
            control.dataset.action === action && control.dataset.key === key);
        (target || dialog.querySelector('[name="memberQuery"]'))?.focus({ preventScroll: true });
      });
    });
  }
  addEventListener("kbot:knowledge-context-changing", () => { reset(); state.libraries = []; state.libraryId = ""; render(); });
  addEventListener("kbot:knowledge-context-restored", () => { if (host) void load(); });
  addEventListener("pagehide", () => { clearInterval(progressTimer); reset(); });
  addEventListener("pageshow", event => {
    if (!event.persisted || !host) return;
    if (view === "processing") progressTimer = setInterval(trackProgress, 6000);
    if (view === "processing" || view === "search") void load();
  });
  shell.ready.then(snapshot => {
    if (!snapshot) return;
    access = snapshot;
    host = document.getElementById("knowledge-owned-workspace");
    if (!host) return;
    reset();
    if (management) {
      host.addEventListener("compositionend", event => {
        if (event.target.name === "filterTitle") { state.filterTitle = event.target.value; filterCatalog(); }
      });
      document.addEventListener("pointerdown", event => {
        if (!event.target.closest(".management-catalog-overflow")) closeCatalogMenus();
      });
      document.addEventListener("keydown", event => { if (event.key === "Escape") closeCatalogMenus(true); });
    } else host.classList.add("knowledge-purpose-page");
    dialog = document.createElement("dialog"); dialog.className = "knowledge-purpose-drawer admin-drawer";
    dialog.setAttribute("aria-labelledby", "knowledge-drawer-title");
    dialog.addEventListener("cancel", event => { event.preventDefault(); void perform(closeDrawer); });
    document.body.append(dialog);
    bind(host); bind(dialog);
    if (view === "processing") progressTimer = setInterval(trackProgress, 6000);
    if (view === "files") {
      dialog.addEventListener("contextmenu", event => {
        if (event.target.matches?.("[data-file-preview-image]")) event.preventDefault();
      });
      dialog.addEventListener("error", event => {
        if (!event.target.matches?.("[data-file-preview-image]")) return;
        clearFilePreview(); state.filePreviewState = "error"; render();
      }, true);
      dialog.addEventListener("dragover", event => {
        const zone = event.target.closest(".knowledge-drop");
        if (!zone) return;
        event.preventDefault();
        if (!state.busy && canUpload(library()) && event.dataTransfer?.types.includes("Files")) {
          zone.setAttribute("data-drag-active", "");
          event.dataTransfer.dropEffect = "copy";
        }
      });
      dialog.addEventListener("dragleave", event => {
        const zone = event.target.closest(".knowledge-drop");
        if (zone && !(event.relatedTarget instanceof Node && zone.contains(event.relatedTarget))) zone.removeAttribute("data-drag-active");
      });
      dialog.addEventListener("drop", event => {
        const zone = event.target.closest(".knowledge-drop");
        if (!zone) return;
        event.preventDefault();
        zone.removeAttribute("data-drag-active");
        selectUploads(event.dataTransfer?.files);
      });
    }
    void load();
  });
})();
