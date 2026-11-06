/* 正式 App 独占资料库工作区：当前授权、图文检索与版本治理。 */
(function () {
  "use strict";
  const api = KBotKnowledgeApi, shell = KBotKnowledgeShell, esc = shell.escapeHtml;
  const searchOnly = document.body.dataset.page === "knowledge";
  const state = { libraries: [], libraryId: "", documents: [], result: null, images: [], previews: {},
    urls: new Set(), uploads: new Map(), epoch: 0, busy: false, page: 1, more: false, error: "",
    text: "", output: "ALL", mode: "RELEVANCE", count: 10, combination: "ANY", title: "", descending: false };
  let host;
  function selected() { return state.libraries.find(value => value.library_id === state.libraryId); }
  function can(operation) { return selected()?.permissions.includes(operation); }
  function reset() {
    state.epoch += 1;
    state.urls.forEach(url => URL.revokeObjectURL(url)); state.urls.clear();
    Object.assign(state, { images: [], previews: {}, result: null, text: "", busy: false, page: 1, error: "" });
  }
  function failure(error) {
    if (error?.name === "AbortError") return;
    state.error = error?.message || "知识操作未完成"; shell.toast(state.error, "error");
  }
  function item(key) { return state.result?.items.find(value => value.key === key); }
  function group(key) { return state.result?.linked.find(value => value.anchor_key === key); }
  function button(action, label, attributes = "") { return `<button type="button" class="secondary-button" data-action="${action}" ${attributes}>${esc(label)}</button>`; }
  function evidence(key) {
    const value = item(key); if (!value) return "";
    return `<section class="knowledge-linked-evidence"><strong>${esc(value.kind)}</strong><p class="knowledge-evidence-text">${esc(value.text || "")}</p>
      ${state.previews[key] ? `<img src="${esc(state.previews[key])}" alt="原图证据">` : ""}
      <small>${esc(JSON.stringify(value.locators))}</small>${button("content", "读取精确内容", `data-key="${esc(key)}"`)}</section>`;
  }
  function render() {
    if (!host) return;
    const library = selected();
    const search = library ? `<p class="knowledge-capability-note">只检索当前 App / Domain / 资料库已发布资料。已配置：
      ${esc((library.search_capabilities || []).map(value => value.direction + " / " + value.kind).join("；"))}。
      NEAR_DUPLICATE 仅为原图字节重复，不代表语义图搜图已配置。</p>
      <form data-search class="knowledge-search-form"><label>检索方式<select name="mode"><option value="RELEVANCE">相关性检索</option><option value="FILTERED_ORDER">全范围筛选排序</option><option value="EXHAUSTIVE">完整枚举</option></select></label>
      <label>文字查询<textarea name="text" rows="2" maxlength="32000" placeholder="输入文字，或仅选择图片">${esc(state.text)}</textarea></label>
      <div class="knowledge-query-images"><label class="secondary-button">选择查询图片<input data-images type="file" accept="image/png,image/jpeg,image/webp" multiple></label>
      ${state.images.map(image => `<img src="${esc(image.url)}" alt="${esc(image.file.name)}">`).join("")}
      ${state.images.length > 1 ? '<select name="combination" aria-label="多图组合"><option value="ANY">任一图</option><option value="ALL">所有图</option><option value="COMPOSED">组合语义</option></select>' : ""}</div>
      <div class="section-toolbar"><label>结果类型<select name="output"><option value="ALL">图文混合</option><option value="TEXT">文字与表格</option><option value="IMAGE">图片与页面</option></select></label>
      <label>主结果数<input name="count" type="number" value="${state.count}" min="1" max="100" required></label><label>排序标题前缀<input name="title" value="${esc(state.title)}"></label>
      <label><input name="descending" type="checkbox" ${state.descending ? "checked" : ""}>标题降序</label><button class="primary-button" ${library.status !== "ACTIVE" ? "disabled" : ""}>${state.busy ? "处理中…" : "检索"}</button></div>
      <p>全范围排序忽略文本和图片，只使用标题条件及显式排序。</p></form>` : '<p>当前 Domain 暂无可访问资料库，请创建资料库或请管理员授予 READ。</p>';
    const results = state.result ? `<section class="knowledge-result-list" aria-live="polite"><p>${esc(state.result.status)} · 主结果 ${state.result.primary_hits.length} · ${esc(state.result.reason || "命中与关联内容分别保留")}</p>
      <details><summary>通道状态</summary><pre>${esc(JSON.stringify(state.result.channels, null, 2))}</pre></details>
      ${state.result.primary_hits.map(hit => `<article class="knowledge-evidence-card"><h3>#${hit.rank}</h3>${evidence(hit.key)}
      ${group(hit.key) ? `<details><summary>引用图片、说明文字与文档关联 · ${esc(group(hit.key).manifest_status)}</summary>
      ${group(hit.key).linked_keys.map(evidence).join("")}
      ${group(hit.key).continuation ? button("relations", "继续读取关联", `data-key="${esc(hit.key)}"`) : ""}</details>` : ""}</article>`).join("")}
      ${state.result.continuation ? button("continue", "下一页主结果") : ""}</section>` : "";
    const docs = !searchOnly && library ? `<section class="knowledge-documents"><header class="section-toolbar"><h2>文档与版本</h2>
      ${can("UPLOAD") ? '<label class="secondary-button">上传原文<input data-upload type="file"></label>' : ""}</header><div class="table-scroll"><table><thead><tr><th>文档</th><th>审核 / 处理 / 发布</th><th>操作</th></tr></thead><tbody>
      ${state.documents.flatMap(document => Object.entries(document.versions).map(([id, version]) => `<tr><td>${esc(document.title)}<small>${esc(id)}</small></td><td>${esc(version.review)} / ${esc(version.processing_status)} / ${esc(version.status)}<small class="table-error">${esc(version.error_code || "")}</small></td><td class="table-actions">
      ${button("source", "原文", `data-version="${esc(id)}"`)}
      ${can("REVIEW") && version.review === "PENDING" ? button("review", "审核通过", `data-version="${esc(id)}" data-decision="APPROVE"`) + button("review", "拒绝", `data-version="${esc(id)}" data-decision="REJECT"`) : ""}
      ${can("PUBLISH") && version.review === "APPROVED" && version.generation_ready && !["PUBLISHED","WITHDRAWN"].includes(version.status) ? button("publish", "发布", `data-version="${esc(id)}"`) : ""}
      ${can("WITHDRAW") && version.status === "PUBLISHED" ? button("withdraw", "撤回", `data-version="${esc(id)}"`) : ""}
      ${can("UPLOAD") && !["PUBLISHED","WITHDRAWN","CANCELLED","FAILED"].includes(version.status) ? button("cancel", "取消任务", `data-version="${esc(id)}"`) : ""}</td></tr>`)).join("") || '<tr><td colspan="3">暂无当前主体可见的文档</td></tr>'}
      </tbody></table></div><footer class="section-toolbar">${state.page > 1 ? button("previous", "上一页") : ""}第 ${state.page} 页${state.more ? button("next", "下一页") : ""}</footer>
      ${can("MANAGE") ? `<details><summary>资料库状态与显式成员授权</summary><p>每个成员至少包含 READ，并保留一个 MANAGE 主体。</p>
      ${button("toggle", library.status === "ACTIVE" ? "停用资料库" : "启用资料库")}<form data-grants><label>成员授权 JSON<textarea name="grants" rows="6">${esc(JSON.stringify(library.grants, null, 2))}</textarea></label><button class="secondary-button">保存授权</button></form></details>` : ""}</section>` : "";
    host.innerHTML = `<header class="section-toolbar"><label>资料库 · 当前 Domain<select data-library>${state.libraries.length ? state.libraries.map(value => `<option value="${esc(value.library_id)}">${esc(value.display_name)} · ${esc(value.status)}</option>`).join("") : '<option value="">暂无资料库</option>'}</select></label>${button("refresh", "刷新状态")}</header>
      ${!searchOnly ? '<details><summary>新建资料库</summary><form data-create class="section-toolbar"><input name="display_name" required maxlength="256" placeholder="资料库名称" aria-label="资料库名称"><button class="primary-button">创建</button></form></details>' : ""}
      <p role="alert" class="table-error">${esc(state.error)}</p>${search}${results}${docs}`;
    host.querySelector("[data-library]").value = state.libraryId;
    for (const name of ["mode", "output", "combination"]) { const node = host.querySelector(`[name="${name}"]`); if (node) node.value = state[name]; }
    if (state.busy) host.querySelectorAll("button,input,select,textarea").forEach(node => { node.disabled = true; });
  }
  async function loadDocuments() {
    if (searchOnly || !state.libraryId) return;
    const epoch = state.epoch;
    const value = await api.request(`/knowledge/libraries/${state.libraryId}/documents?page=${state.page}&page_size=20`);
    if (epoch !== state.epoch) return;
    state.documents = value.items; state.more = value.has_more;
  }
  async function load() {
    reset(); const epoch = state.epoch;
    state.libraries = []; state.documents = []; state.libraryId = ""; render();
    const values = await api.request("/knowledge/libraries");
    if (epoch !== state.epoch) return;
    state.libraries = values; state.libraryId = values[0]?.library_id || "";
    await loadDocuments(); render();
  }
  async function perform(work) {
    if (state.busy) return;
    state.busy = true; state.error = ""; const epoch = state.epoch; render();
    try { await work(); } catch (error) { if (epoch === state.epoch) failure(error); }
    finally { if (epoch === state.epoch) { state.busy = false; render(); } }
  }
  async function imageBase64(file) {
    const raw = new Uint8Array(await file.arrayBuffer()); let binary = "";
    for (let index = 0; index < raw.length; index += 32768) binary += String.fromCharCode(...raw.subarray(index, index + 32768));
    return btoa(binary);
  }
  async function search() {
    const epoch = state.epoch, structured = state.mode !== "RELEVANCE";
    const images = structured ? [] : await Promise.all(state.images.map(async image => ({ mime_type: image.file.type, image_base64: await imageBase64(image.file) })));
    if (epoch !== state.epoch) return;
    const result = await api.json(`/knowledge/libraries/${state.libraryId}/search`, "POST", {
      text: structured ? null : state.text.trim() || null, images, requested_count: state.count,
      output_types: state.output === "ALL" ? ["TEXT","FIGURE","PHOTO","TABLE","PAGE"] : state.output === "IMAGE" ? ["FIGURE","PHOTO","PAGE","REGION"] : ["TEXT","TABLE"],
      combination: images.length > 1 ? state.combination : null, mode: state.mode,
      predicates: structured && state.title ? [{ field: "title", operator: "PREFIX", value: state.title }] : [],
      order_by: structured ? [{ field: "title", descending: state.descending }] : [],
    });
    if (epoch === state.epoch) state.result = result;
  }
  async function content(key) {
    const value = item(key), epoch = state.epoch;
    if (!value) return;
    const blob = await api.requestBlob(`/knowledge/retrievals/${value.content_ref.retrieval_ref}/content/${key}`);
    if (epoch !== state.epoch) return;
    const url = URL.createObjectURL(blob); state.urls.add(url);
    if (/^image\/(png|jpeg|webp)$/.test(blob.type)) state.previews[key] = url;
    else download(url, "知识证据");
  }
  function download(url, name) { const link = document.createElement("a"); link.href = url; link.download = name; link.click(); }
  async function act(button) {
    const action = button.dataset.action, version = button.dataset.version, library = selected(), epoch = state.epoch;
    if (action === "refresh") return load();
    if (action === "content") return content(button.dataset.key);
    if (action === "next" || action === "previous") { state.page += action === "next" ? 1 : -1; return loadDocuments(); }
    if (action === "continue" || action === "relations") {
      const related = group(button.dataset.key);
      const result = await api.json(`/knowledge/retrievals/${state.result.retrieval_ref}${related ? "/relations:continue" : ":continue"}`, "POST",
        related ? { anchor_key: related.anchor_key, cursor: related.continuation } : { cursor: state.result.continuation });
      if (epoch === state.epoch) state.result = result;
      return;
    }
    if (action === "source") {
      const blob = await api.requestBlob(`/knowledge/libraries/${state.libraryId}/versions/${version}/source`);
      if (epoch !== state.epoch) return;
      const url = URL.createObjectURL(blob); state.urls.add(url); download(url, "资料原文"); return;
    }
    if (action === "toggle") {
      await api.json(`/knowledge/libraries/${state.libraryId}`, "PATCH", { expected_revision: library.row_version, status: library.status === "ACTIVE" ? "DISABLED" : "ACTIVE" });
      return load();
    }
    if (action === "withdraw" && !await shell.confirmAction({ title: "撤回资料？", message: "撤回后当前检索和历史引用均不能继续读取。", confirmLabel: "撤回", tone: "danger" })) return;
    await api.json(`/knowledge/libraries/${state.libraryId}/versions/${version}:${action}`, "POST",
      button.dataset.decision ? { decision: button.dataset.decision } : undefined);
    await loadDocuments();
  }
  function bind() {
    host.addEventListener("input", event => {
      const name = event.target.name;
      if (["text","title","mode","output","combination"].includes(name)) state[name] = event.target.value;
      if (name === "count") state.count = Number(event.target.value);
      if (name === "descending") state.descending = event.target.checked;
    });
    host.addEventListener("submit", event => {
      event.preventDefault(); const form = event.target, values = new FormData(form);
      void perform(async () => {
        if (form.matches("[data-search]")) return search();
        if (form.matches("[data-create]")) { await api.json("/knowledge/libraries", "POST", { display_name: values.get("display_name") }); return load(); }
        if (form.matches("[data-grants]")) {
          await api.json(`/knowledge/libraries/${state.libraryId}`, "PATCH", { expected_revision: selected().row_version, grants: JSON.parse(values.get("grants")) });
          return load();
        }
      });
    });
    host.addEventListener("click", event => { const button = event.target.closest("[data-action]"); if (button) void perform(() => act(button)); });
    host.addEventListener("change", event => {
      if (event.target.matches("[data-library]")) {
        const id = event.target.value; reset(); state.libraryId = id; state.documents = [];
        void perform(loadDocuments); return;
      }
      const files = Array.from(event.target.files || []);
      if (event.target.matches("[data-images]")) {
        if (files.length > 4 || files.reduce((size, file) => size + file.size, 0) > 8 * 1024 * 1024 || files.some(file => !["image/png","image/jpeg","image/webp"].includes(file.type))) { failure(new Error("最多 4 张 PNG/JPEG/WebP，合计不超过 8 MiB")); render(); return; }
        state.images.forEach(image => { URL.revokeObjectURL(image.url); state.urls.delete(image.url); });
        state.images = files.map(file => { const url = URL.createObjectURL(file); state.urls.add(url); return { file, url }; }); render();
      }
      if (event.target.matches("[data-upload]") && files[0]) {
        const file = files[0], identity = [state.libraryId,file.name,file.size,file.lastModified].join(":");
        if (file.size > 64 * 1024 * 1024) { failure(new Error("文件超过 64 MiB")); return; }
        const key = state.uploads.get(identity) || api.requestId(); state.uploads.set(identity, key);
        void perform(async () => {
          await api.request(`/knowledge/libraries/${state.libraryId}/documents?name=${encodeURIComponent(file.name)}`, { method: "POST", headers: { "Content-Type": file.type || "application/octet-stream", "Idempotency-Key": key }, body: file });
          state.uploads.delete(identity); await loadDocuments(); shell.toast("已登记解析任务；索引就绪并审核通过后才可发布");
        });
      }
    });
  }
  addEventListener("kbot:knowledge-context-changing", () => { reset(); state.libraries = []; state.documents = []; state.libraryId = ""; render(); });
  addEventListener("kbot:knowledge-context-restored", () => { void load().catch(error => { failure(error); render(); }); });
  addEventListener("pagehide", reset);
  shell.ready.then(async access => {
    if (!access) return;
    host = document.getElementById("knowledge-owned-workspace");
    if (!host) return;
    bind(); try { await load(); } catch (error) { failure(error); render(); }
  }).catch(() => {});
})();
