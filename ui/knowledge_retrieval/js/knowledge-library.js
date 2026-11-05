/* 资料工作区：上传、审核、解析与文件目录复用同一 KC 事实和当前会话边界。 */
(function () {
  "use strict";
  const { escapeHtml: esc, toast, badge } = KBotKnowledgeShell;
  const $ = id => document.getElementById(id);
  const page = document.body.dataset.page;
  const state = { collections: [], rows: [], page: 1, size: 20, total: 0, request: 0, timer: null, busy: false, entries: [], rejection: null };
  const list = value => Array.isArray(value) ? value : value?.collections || value?.items || [];
  const collectionId = () => $("library-collection").value;
  const root = () => "/knowledge-cores/" + encodeURIComponent(collectionId());
  const bundlePath = row => root() + "/bundles/" + encodeURIComponent(row.bundle_id) + "/revisions/" + encodeURIComponent(row.bundle_revision_id);
  const report = error => toast(error.message || "无法完成资料操作，请重试。", "error");
  const time = value => value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "—";
  const size = value => value == null ? "—" : value < 1024 ? value + " B" : value < 1048576 ? (value / 1024).toFixed(1) + " KB" : (value / 1048576).toFixed(1) + " MB";
  const labels = { APPROVAL_PENDING:"待审核", PENDING:"等待处理", QUEUED:"排队中", PARSE_PENDING:"等待解析", PARSING:"解析中", INDEXING:"建立索引", PROFILE_PENDING:"等待画像", PROFILING:"生成画像", VISUAL_ENHANCING:"可检索 · 视觉增强中", DISCOVERY_INDEXING:"建立检索索引", READY:"可检索", AVAILABLE:"可检索", PARTIAL:"部分可用", FAILED:"失败", CANCELLED:"已取消", SOURCE_UNAVAILABLE:"源文件不可用", RECEIVED:"等待解析" };
  const status = value => badge(labels[value] || value || "未知", ["READY","AVAILABLE"].includes(value) ? "good" : ["FAILED","SOURCE_UNAVAILABLE"].includes(value) ? "bad" : "warn");
  const rowEmpty = message => '<tr><td colspan="' + (page === "files" ? 6 : 5) + '" class="empty-cell">' + esc(message) + '</td></tr>';
  const button = (action, id, label, danger = false) => '<button class="' + (danger ? 'danger-row-action' : 'row-action') + '" type="button" data-library-action="' + action + '" data-row="' + id + '">' + label + '</button>';

  function lock(busy) {
    state.busy = busy;
    document.querySelectorAll("main input,main select,main button,#library-reject button,#library-reject textarea").forEach(node => { node.disabled = busy; });
    if (page === "upload") renderUpload();
  }

  function renderRows() {
    const query = ($("library-query")?.value || "").trim().toLowerCase();
    const rows = page === "reviews" ? state.rows.filter(row => [row.title, ...(row.document_names || [])].join(" ").toLowerCase().includes(query)) : state.rows;
    if ($("library-result-count")) $("library-result-count").textContent = "显示 " + rows.length + " / " + state.rows.length + " 项待审核资料";
    $("library-rows").innerHTML = rows.map(row => {
      const id = state.rows.indexOf(row);
      if (page === "reviews") return '<tr><td><strong class="table-primary">' + esc(row.title) + '</strong></td><td><div class="review-file-list">' + (row.document_names || []).slice(0,2).map(name => '<span class="tag">' + esc(name) + '</span>').join("") + ((row.document_names || []).length > 2 ? '<small>另有 ' + (row.document_names.length - 2) + ' 个文件</small>' : '') + '</div></td><td>' + badge("待审核","warn") + '</td><td>' + badge("等待处理") + '</td><td><div class="table-actions review-row-actions">' + button("preview",id,"预览") + button("approve",id,"通过") + button("reject",id,"退回",true) + '</div></td></tr>';
      if (page === "files") return '<tr><td><strong class="table-primary">' + esc(row.file_name) + '</strong>' + (row.failure_message ? '<small class="table-error">' + esc(row.failure_message) + '</small>' : '') + '</td><td>' + esc(row.bundle_title) + '<small class="table-secondary">版本 ' + esc(row.revision_no) + '</small></td><td>' + esc(row.mime_type || "类型未知") + '<small class="table-secondary">' + size(row.byte_size) + '</small></td><td>' + status(row.status) + '</td><td>' + time(row.received_at) + '</td><td>' + button("preview",id,"查看详情") + '</td></tr>';
      const progress = Math.max(0, Math.min(100, Number(row.progress_percent) || 0));
      return '<tr><td><strong class="table-primary">' + esc(row.title) + '</strong><small class="table-secondary">版本 ' + esc(row.revision_no || 1) + '</small></td><td>' + status(row.current_stage || row.status) + '</td><td><div class="processing-progress"><span><strong>' + esc(row.ready_count || 0) + '</strong> / ' + esc(row.file_count || 0) + ' 可用' + (row.failed_count ? ' · ' + esc(row.failed_count) + ' 失败' : '') + '</span><i><b style="width:' + progress + '%"></b></i></div></td><td>' + time(row.reviewed_at) + '</td><td>' + button("detail",id,"查看详情") + '</td></tr>';
    }).join("") || rowEmpty(page === "reviews" ? "当前条件下没有待审核资料。" : "当前条件下没有" + (page === "files" ? "知识库文件。" : "解析任务。"));
    if ($("library-pagination")) {
      $("library-pagination").hidden = !state.total;
      $("library-page-summary").textContent = "第 " + ((state.page - 1) * state.size + 1) + "–" + Math.min(state.page * state.size,state.total) + " 项，共 " + state.total + " 项";
      $("library-page").textContent = state.page;
      $("library-prev").disabled = state.page === 1;
      $("library-next").disabled = state.page * state.size >= state.total;
    }
  }

  async function loadRows() {
    clearTimeout(state.timer);
    const request = ++state.request;
    const selected = state.collections.find(row => String(row.collection_id) === collectionId());
    if ($("library-collection-title")) $("library-collection-title").textContent = selected?.display_name || "当前知识库";
    if (!selected) { state.rows=[]; state.total=0; renderRows(); return; }
    $("library-rows").innerHTML = rowEmpty("正在读取…");
    try {
      const path = page === "reviews" ? root() + "/approvals" : KBotKnowledgeApi.withQuery(root() + "/" + page, { page:state.page, page_size:state.size, q:$("library-query")?.value.trim(), status:$("library-status")?.value });
      const result = await KBotKnowledgeApi.json(path,"GET");
      if (request !== state.request) return;
      state.rows = list(result); state.total = result.total ?? state.rows.length;
      renderRows();
      if (page === "processing" && state.rows.some(row => !["READY","PARTIAL","FAILED","CANCELLED"].includes(row.status))) state.timer = setTimeout(() => loadRows().catch(report),10000);
    } catch (error) {
      if (request === state.request) {
        state.rows=[]; state.total=0; renderRows();
        $("library-rows").innerHTML=rowEmpty(error.message || "读取失败，请点击查询或刷新重试。");
      }
      throw error;
    }
  }

  async function review(row, decision, comment = null) {
    if (state.busy) return;
    lock(true);
    try {
      await KBotKnowledgeApi.json(root() + "/bundle-revisions/" + encodeURIComponent(row.bundle_revision_id) + "/approval","POST",{ decision, comment });
      $("library-reject").close(); state.rejection=null;
      await loadRows();
      toast(decision === "APPROVE" ? "资料已通过，进入解析与索引流程。" : "资料已退回。");
    } finally { lock(false); }
  }

  function showDetail(row) {
    $("library-detail-title").textContent=row.title || row.file_name || "资料详情";
    $("library-detail-body").innerHTML='<div class="processing-stage-strip">' + ["审核通过","解析","索引","可检索"].map(label=>'<span>' + label + '</span>').join("") + '</div><section class="key-value-card"><dl class="key-value-table"><div><dt>当前阶段</dt><dd>' + status(row.current_stage || row.status) + '</dd></div><div><dt>审核通过</dt><dd>' + time(row.reviewed_at) + '</dd></div><div><dt>处理完成</dt><dd>' + time(row.completed_at) + '</dd></div></dl></section><section class="table-card processing-file-table"><table><thead><tr><th>文件</th><th>状态</th><th>操作</th></tr></thead><tbody>' + (row.files || []).map((file,index)=>'<tr><td>' + esc(file.name) + (file.failure_message ? '<small class="table-error">' + esc(file.failure_message) + '</small>' : '') + '</td><td>' + status(file.status) + '</td><td>' + (file.status === "FAILED" && file.document_version_id ? '<button type="button" class="row-action" data-reprocess="' + index + '">重新解析</button>' : '—') + '</td></tr>').join("") + '</tbody></table></section>';
    $("library-detail-body").querySelectorAll("[data-reprocess]").forEach(node => node.addEventListener("click",async () => {
      if (!await KBotKnowledgeShell.confirmAction({ title: "重新解析文件？", message: "确认重新解析这个失败文件？", confirmLabel: "重新解析", tone: "warning" })) return;
      node.disabled=true;
      try {
        await KBotKnowledgeApi.json(bundlePath(row) + "/reprocess","POST",{document_version_id:row.files[Number(node.dataset.reprocess)].document_version_id});
        $("library-detail").close(); await loadRows(); toast("已提交重新解析。");
      } catch(error) { report(error); } finally { node.disabled=false; }
    }));
    $("library-detail").showModal();
  }

  async function preview(row) {
    const selected=collectionId();
    const result=await KBotKnowledgeApi.json(bundlePath(row) + "/preview","GET");
    if (selected !== collectionId()) return;
    $("library-detail-title").textContent=row.file_name || result.title || "资料预览";
    const files = (result.files || []).filter(file => !row.document_version_id || String(file.document_version_id) === String(row.document_version_id));
    $("library-detail-body").innerHTML='<p>原始文件由当前业务空间授权访问；文件不可用时不会提供下载入口。</p><section class="table-card"><div class="table-scroll"><table><thead><tr><th>文件</th><th>类型 / 大小</th><th>状态</th><th>操作</th></tr></thead><tbody>' + files.map((file,index)=>'<tr><td>' + esc(file.declared_name || file.external_document_id || "未命名文件") + '</td><td>' + esc(file.detected_mime_type || file.declared_mime_type || "未知") + '<small class="table-secondary">' + size(file.byte_size) + '</small></td><td>' + esc(file.member_status || "—") + '</td><td>' + (file.preview_available && file.document_version_id ? '<button class="row-action" type="button" data-content="' + index + '">打开原始文件</button>' : '源文件不可用') + '</td></tr>').join("") + '</tbody></table></div></section>';
    $("library-detail-body").querySelectorAll("[data-content]").forEach(node => node.addEventListener("click",async () => {
      const file=files[Number(node.dataset.content)];
      const popup=window.open("about:blank","_blank");
      if (popup) popup.opener=null;
      node.disabled=true;
      try {
        const blob=await KBotKnowledgeApi.requestBlob(bundlePath(row) + "/documents/" + encodeURIComponent(file.document_version_id) + "/content");
        KBotKnowledgeApi.presentBlob(blob,file.declared_name || "资料文件",popup);
      } catch(error) { popup?.close(); report(error); } finally { node.disabled=false; }
    }));
    $("library-detail").showModal();
  }

  function grouping() { return document.querySelector('input[name="grouping"]:checked').value; }
  function renderUpload() {
    const bundle=grouping() === "SINGLE_BUNDLE";
    $("upload-bundle-field").hidden=!bundle;
    document.querySelectorAll(".upload-mode-option").forEach(node=>node.classList.toggle("selected",node.querySelector("input").checked));
    $("upload-ledger").hidden=!state.entries.length;
    $("upload-summary").textContent=state.entries.length + " 个 · " + size(state.entries.reduce((total,entry)=>total + entry.file.size,0));
    $("upload-submit-summary").textContent=bundle ? "提交为 1 个 Bundle" : "提交为 " + state.entries.length + " 个独立知识对象";
    $("upload-submit").disabled=state.busy || !collectionId() || !state.entries.length;
    $("upload-submit").textContent=state.busy ? "正在校验并提交…" : "提交资料";
    $("upload-files").innerHTML=state.entries.map((entry,index)=>'<article class="upload-file-row"><span class="upload-file-order">' + (index+1) + '</span><div class="upload-file-identity"><strong>' + esc(entry.file.name) + '</strong><small>' + esc(entry.file.type || "未知类型") + ' · ' + size(entry.file.size) + '</small></div>' + (bundle ? '<div class="upload-file-settings"><label class="control-field compact-control-field"><span class="field-label">文件角色</span><select data-file-role="' + index + '"><option value="CONTENT"' + (entry.role === "CONTENT" ? " selected" : "") + '>正文内容</option><option value="SUPPLEMENT"' + (entry.role === "SUPPLEMENT" ? " selected" : "") + '>补充材料</option></select></label><label class="upload-required-toggle"><input type="checkbox" data-file-required="' + index + '"' + (entry.required ? " checked" : "") + '><span>必要成员</span></label></div>' : '<span class="status-pill">独立资料</span>') + '<button class="row-action danger-row-action" type="button" data-remove-file="' + index + '">移除</button></article>').join("");
    $("upload-files").querySelectorAll("input,select,button").forEach(node=>{ node.disabled=state.busy; });
  }
  function appendFiles(files) {
    if (state.busy) return;
    for (const file of files) state.entries.push({ id:KBotKnowledgeApi.requestId(), file, role:"CONTENT", required:true });
    $("upload-receipt").hidden=true; renderUpload();
  }
  function sha256Fallback(bytes) {
    const constants = [
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
      0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
      0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
      0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
      0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
      0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
      0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
      0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
      0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    const state = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    const paddedLength = Math.ceil((bytes.length + 9) / 64) * 64;
    const padded = new Uint8Array(paddedLength);
    padded.set(bytes);
    padded[bytes.length] = 0x80;
    new DataView(padded.buffer).setUint32(paddedLength - 4, bytes.length * 8);
    const rotate = (value, bits) => (value >>> bits) | (value << (32 - bits));
    for (let offset = 0; offset < padded.length; offset += 64) {
      const schedule = new Uint32Array(64);
      const view = new DataView(padded.buffer, offset, 64);
      for (let index = 0; index < 16; index += 1) schedule[index] = view.getUint32(index * 4);
      for (let index = 16; index < 64; index += 1) {
        const a = schedule[index - 15];
        const b = schedule[index - 2];
        const smallSigma0 = rotate(a, 7) ^ rotate(a, 18) ^ (a >>> 3);
        const smallSigma1 = rotate(b, 17) ^ rotate(b, 19) ^ (b >>> 10);
        schedule[index] = (schedule[index - 16] + smallSigma0 + schedule[index - 7] + smallSigma1) >>> 0;
      }
      let [a, b, c, d, e, f, g, h] = state;
      for (let index = 0; index < 64; index += 1) {
        const bigSigma1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
        const choose = (e & f) ^ (~e & g);
        const temp1 = (h + bigSigma1 + choose + constants[index] + schedule[index]) >>> 0;
        const bigSigma0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
        const majority = (a & b) ^ (a & c) ^ (b & c);
        const temp2 = (bigSigma0 + majority) >>> 0;
        [h, g, f, e, d, c, b, a] = [g, f, e, (d + temp1) >>> 0, c, b, a, (temp1 + temp2) >>> 0];
      }
      state[0] = (state[0] + a) >>> 0; state[1] = (state[1] + b) >>> 0;
      state[2] = (state[2] + c) >>> 0; state[3] = (state[3] + d) >>> 0;
      state[4] = (state[4] + e) >>> 0; state[5] = (state[5] + f) >>> 0;
      state[6] = (state[6] + g) >>> 0; state[7] = (state[7] + h) >>> 0;
    }
    return state.map((value) => value.toString(16).padStart(8, "0")).join("");
  }

  async function sha256(file) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (globalThis.crypto?.subtle?.digest) {
      const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
      return Array.from(new Uint8Array(digest)).map((value) => value.toString(16).padStart(2, "0")).join("");
    }
    return sha256Fallback(bytes);
  }

  async function upload(event) {
    event.preventDefault();
    if (state.busy || !collectionId() || !state.entries.length) return;
    lock(true);
    try {
      const body=new FormData();
      const declarations=await Promise.all(state.entries.map(async(entry,index)=>({ part_name:"file_" + index,client_file_id:entry.id,display_name:entry.file.name,declared_mime_type:entry.file.type || "application/octet-stream",byte_size:entry.file.size,content_sha256:await sha256(entry.file),ordinal:index,role:grouping() === "SINGLE_BUNDLE" ? entry.role : "CONTENT",required_flag:grouping() === "SINGLE_BUNDLE" ? entry.required : false })));
      body.set("grouping_mode",grouping());
      body.set("files",JSON.stringify(declarations));
      const generatedName=new Date().toISOString().replace(/[^0-9]/g,"").slice(0,12);
      body.set("bundle_name",generatedName);
      if (grouping() === "SINGLE_BUNDLE") body.set("bundle",JSON.stringify({client_bundle_id:KBotKnowledgeApi.requestId(),title:$("upload-bundle-title").value.trim() || generatedName,security_level:1}));
      state.entries.forEach((entry,index)=>body.append("file_" + index,entry.file,entry.file.name));
      const result=await KBotKnowledgeApi.request(root() + "/ingestions/user-files",{method:"POST",body,headers:{"Idempotency-Key":KBotKnowledgeApi.requestId()}});
      if (!Array.isArray(result?.items)) throw new Error("上传响应缺少受理结果，请检查审核队列后再决定是否重试。");
      const rejected=result.items.filter(row=>row.status === "REJECTED");
      const receipt=$("upload-receipt");
      receipt.hidden=false; receipt.className="notice upload-receipt " + (rejected.length ? "notice-warning" : "notice-success");
      receipt.innerHTML='<strong>' + (rejected.length ? "部分资料未受理" : "资料已进入审核队列") + '</strong><span>已接收 ' + (result.items.length-rejected.length) + ' 项，未受理 ' + rejected.length + ' 项 · 请求编号 ' + esc(result.request_id || "—") + '</span>' + rejected.map(row=>'<p>' + esc(row.message || "文件未通过受理，请检查后重试。") + '</p>').join("");
      if (!rejected.length) { state.entries=[]; $("upload-bundle-title").value=""; }
      else if (grouping() === "EACH_FILE" && rejected.every(row=>row.client_file_id)) {
        const ids=new Set(rejected.map(row=>row.client_file_id));
        state.entries=state.entries.filter(entry=>ids.has(entry.id));
      }
    } finally { lock(false); }
  }

  function bindUpload() {
    $("upload-choose").addEventListener("click",()=>$("upload-picker").click());
    $("upload-picker").addEventListener("change",event=>{appendFiles(event.target.files);event.target.value="";});
    $("upload-clear").addEventListener("click",()=>{if(!state.busy){state.entries=[];renderUpload();}});
    $("upload-files").addEventListener("click",event=>{
      const node=event.target.closest("[data-remove-file]");
      if(node && !state.busy){state.entries.splice(Number(node.dataset.removeFile),1);renderUpload();}
    });
    $("upload-files").addEventListener("change",event=>{
      const node=event.target;
      if (node.dataset.fileRole !== undefined) state.entries[Number(node.dataset.fileRole)].role=node.value;
      if (node.dataset.fileRequired !== undefined) state.entries[Number(node.dataset.fileRequired)].required=node.checked;
    });
    document.querySelectorAll('input[name="grouping"]').forEach(node=>node.addEventListener("change",()=>{$("upload-receipt").hidden=true;renderUpload();}));
    const zone=$("upload-drop-zone");
    for(const name of ["dragenter","dragover"]) zone.addEventListener(name,event=>{event.preventDefault();zone.classList.add("dragging");});
    zone.addEventListener("dragleave",()=>zone.classList.remove("dragging"));
    zone.addEventListener("drop",event=>{event.preventDefault();zone.classList.remove("dragging");appendFiles(event.dataTransfer.files);});
    $("library-upload-form").addEventListener("submit",event=>upload(event).catch(report));
  }

  KBotKnowledgeShell.ready.then(async access=>{
    if(!access)return;
    if(page === "upload") bindUpload();
    else {
      $("library-refresh").addEventListener("click",()=>{state.page=1;loadRows().catch(report);});
      $("library-query")?.addEventListener("input",()=>{if(page === "reviews")renderRows();});
      $("library-query")?.addEventListener("keydown",event=>{if(event.key==="Enter"){event.preventDefault();state.page=1;loadRows().catch(report);}});
      $("library-status")?.addEventListener("change",()=>{state.page=1;loadRows().catch(report);});
      $("library-page-size")?.addEventListener("change",event=>{state.size=Number(event.target.value);state.page=1;loadRows().catch(report);});
      for(const [id,delta] of [["library-prev",-1],["library-next",1]]) $(id)?.addEventListener("click",()=>{state.page+=delta;loadRows().catch(report);});
      $("library-rows").addEventListener("click",event=>{
        const node=event.target.closest("[data-library-action]");if(!node || state.busy)return;
        const row=state.rows[Number(node.dataset.row)];
        if(node.dataset.libraryAction === "detail")showDetail(row);
        if(node.dataset.libraryAction === "preview")preview(row).catch(report);
        if(node.dataset.libraryAction === "approve")review(row,"APPROVE").catch(report);
        if(node.dataset.libraryAction === "reject"){
          state.rejection=row;$("library-reject-title").textContent=row.title;
          $("library-reject-reason").value="";$("library-reject").showModal();
        }
      });
      $("library-reject-form").addEventListener("submit",event=>{event.preventDefault();const reason=$("library-reject-reason").value.trim();if(reason && state.rejection)review(state.rejection,"REJECT",reason).catch(report);});
      document.querySelectorAll("[data-close-dialog]").forEach(node=>node.addEventListener("click",()=>{if(!state.busy)$(node.dataset.closeDialog).close();}));
      $("library-reject").addEventListener("cancel",event=>{if(state.busy)event.preventDefault();});
    }
    $("library-collection").addEventListener("change",()=>{
      state.page=1;
      if(page === "upload"){$("upload-receipt").hidden=true;renderUpload();}
      else {for(const id of ["library-detail","library-reject"])$(id).close();loadRows().catch(report);}
      const url=new URL(location.href);url.searchParams.set("collection_id",collectionId());history.replaceState(null,"",url);
    });
    try {
      state.collections=list(await KBotKnowledgeApi.json("/knowledge-cores","GET"));
      if(page === "upload")state.collections=state.collections.filter(row=>row.status === "ACTIVE");
      if (!state.collections.length) {
        const titles = {upload:"没有可贡献的知识库",reviews:"没有可审核的知识库",processing:"没有可跟踪的知识库",files:"知识库中还没有文件"};
        const content = document.querySelector("main > .table-card,main > .knowledge-upload-workbench");
        content.hidden = true;
        content.insertAdjacentHTML("beforebegin", '<section class="empty-state"><h2>' + titles[page] + '</h2><p>当前业务空间没有可用知识库，请联系知识检索管理员。</p></section>');
      }
      $("library-collection").innerHTML=state.collections.map(row=>'<option value="' + esc(row.collection_id) + '">' + esc(row.display_name) + '</option>').join("") || '<option value="">当前业务空间没有可用知识库</option>';
      const requested=new URL(location.href).searchParams.get("collection_id");
      if(state.collections.some(row=>String(row.collection_id)===requested))$("library-collection").value=requested;
      if(page === "upload")renderUpload();else await loadRows();
    } catch(error) { report(error); if(page !== "upload")$("library-rows").innerHTML=rowEmpty("读取失败，请刷新页面重试。"); }
  });
  addEventListener("pagehide",()=>clearTimeout(state.timer));
})();
