/* Knowledge Core 生命周期页：创建、修改、启停与删除，隔离边界只来自当前 Token。 */
(function () {
  "use strict";

  const { badge, escapeHtml, toast } = KBotAssistantShell;
  let rows = [];
  let catalog = [];
  let editingId = null;
  let selectedCollectionId = null;
  let processingTimer = null;

  function openDialog(id) { document.getElementById(id)?.showModal(); }
  function closeDialog(id) { document.getElementById(id)?.close(); }
  function session() { return globalThis.KBotAssistantAuth?.load?.() || {}; }

  function emptyRow(columns, title, copy) {
    return `<tr><td colspan="${columns}"><div class="assistant-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div></td></tr>`;
  }

  function statusTone(status) {
    if (status === "ACTIVE") return "good";
    if (status === "DISABLED" || status === "DELETING") return "warn";
    return "warn";
  }

  function processingStatusLabel(status) {
    return ({
      APPROVAL_PENDING: "待审核",
      PENDING: "等待处理",
      PARSE_PENDING: "等待解析",
      PARSING: "解析中",
      INDEXING: "索引中",
      PROFILE_PENDING: "等待画像",
      PROFILING: "生成画像",
      DISCOVERY_INDEXING: "建立 Discovery",
      READY: "已完成",
      PARTIAL: "部分完成",
      AVAILABLE: "可用",
      QUEUED: "排队中",
      FAILED: "失败",
      CANCELLED: "已取消",
    })[status] || status || "未知";
  }

  function selectedRow() {
    return rows.find((row) => String(row.collection_id) === String(selectedCollectionId));
  }

  function renderProcessing(items) {
    const node = document.getElementById("kc-processing-rows");
    if (!node) return;
    if (!selectedCollectionId) {
      node.innerHTML = emptyRow(6, "尚未选择 Knowledge Core", "选择一个 Knowledge Core 后，可上传资料并查看解析、索引和 Discovery 进度。");
      return;
    }
    if (!items.length) {
      node.innerHTML = emptyRow(6, "当前没有资料处理记录", "点击“上传文件”提交 PDF、Word 或文本资料。");
      return;
    }
    node.innerHTML = items.map((item) => {
      const status = String(item.status || "");
      const progress = Number.isFinite(Number(item.progress_percent)) ? `${item.progress_percent}%` : "—";
      const fileSummary = `${item.ready_count || 0}/${item.file_count || 0} 个文件完成`;
      return `<tr>
        <td><strong>${escapeHtml(item.title || "未命名资料包")}</strong><br><small>${escapeHtml(item.bundle_id || "")}</small></td>
        <td>${badge(processingStatusLabel(status), statusTone(status))}</td>
        <td>${escapeHtml(item.current_stage || "—")}</td>
        <td>${escapeHtml(progress)}</td>
        <td>${escapeHtml(fileSummary)}${item.failed_count ? ` · ${escapeHtml(item.failed_count)} 个失败` : ""}</td>
        <td>${escapeHtml(item.completed_at || item.reviewed_at || "处理中")}</td>
      </tr>`;
    }).join("");
  }

  async function loadProcessing() {
    const title = document.getElementById("kc-assets-title");
    const upload = document.getElementById("kc-upload-open");
    const refresh = document.getElementById("kc-refresh-processing");
    const row = selectedRow();
    if (title) title.textContent = row ? `${row.display_name || "Knowledge Core"} · 资料处理记录` : "先在上方选择一个 Knowledge Core。";
    if (upload) upload.disabled = !row || row.status === "DISABLED" || row.status === "DELETING";
    if (refresh) refresh.disabled = !row;
    if (!row) {
      renderProcessing([]);
      return;
    }
    const payload = await KBotAssistantApi.json(`/knowledge-cores/${row.collection_id}/processing`, "GET");
    renderProcessing(Array.isArray(payload?.items) ? payload.items : []);
    const active = (payload?.items || []).some((item) => !["READY", "PARTIAL", "FAILED", "CANCELLED"].includes(String(item.status || "")));
    if (active && !processingTimer) {
      processingTimer = globalThis.setTimeout(() => {
        processingTimer = null;
        loadProcessing().catch((error) => toast(error.message || "无法刷新解析进度", "error"));
      }, 3000);
    }
  }

  function collectionItems(payload) {
    if (Array.isArray(payload)) return payload;
    if (Array.isArray(payload?.collections)) return payload.collections;
    if (Array.isArray(payload?.items)) return payload.items;
    return [];
  }

  function modelsOf(row) {
    return row?.models || row?.models_json || {};
  }

  function modelsByCategory(category) {
    return catalog.filter((row) => KBotAssistantApi.modelCategory(row) === category);
  }

  function modelLabel(modelId) {
    if (!modelId) return "—";
    const row = catalog.find((item) => String(item.model_id) === String(modelId));
    return row?.display_name || row?.served_model_name || modelId;
  }

  function fillSelect(select, items, currentId, emptyLabel) {
    const options = [`<option value="">${escapeHtml(emptyLabel)}</option>`].concat(
      items.map((row) => {
        const selected = String(row.model_id) === String(currentId || "") ? " selected" : "";
        const label = row.display_name || row.served_model_name || row.model_id;
        return `<option value="${escapeHtml(row.model_id)}"${selected}>${escapeHtml(label)}</option>`;
      }),
    );
    select.innerHTML = options.join("");
    select.disabled = false;
  }

  function currentDomainLabel() {
    const current = session();
    return current.domain_name || current.domain_id || "当前 Domain";
  }

  function resetForm() {
    editingId = null;
    document.getElementById("kc-dialog-title").textContent = "创建 Knowledge Core";
    document.getElementById("kc-submit").textContent = "创建";
    document.getElementById("kc-form")?.reset();
    document.getElementById("kc-id").value = "";
    document.getElementById("kc-row-version").value = "";
    document.getElementById("kc-domain").value = currentDomainLabel();
    const textEmbeddings = modelsByCategory(KBotAssistantApi.ModelCategory.TXT_EMBEDDING);
    fillSelect(
      document.getElementById("kc-embedding"),
      textEmbeddings,
      "",
      textEmbeddings.length ? "选择文本 Embedding" : "当前没有可用的文本 Embedding",
    );
    fillSelect(
      document.getElementById("kc-visual-embedding"),
      modelsByCategory(KBotAssistantApi.ModelCategory.IMG_EMBEDDING),
      "",
      "不绑定",
    );
  }

  function fillForm(row) {
    editingId = row.collection_id;
    const models = modelsOf(row);
    document.getElementById("kc-dialog-title").textContent = "编辑 Knowledge Core";
    document.getElementById("kc-submit").textContent = "保存";
    document.getElementById("kc-id").value = row.collection_id;
    document.getElementById("kc-row-version").value = row.row_version;
    document.getElementById("kc-domain").value = currentDomainLabel();
    document.getElementById("kc-name").value = row.display_name || "";
    document.getElementById("kc-description").value = row.description || "";
    document.getElementById("kc-security").value = String(row.default_security_level ?? 1);
    fillSelect(
      document.getElementById("kc-embedding"),
      modelsByCategory(KBotAssistantApi.ModelCategory.TXT_EMBEDDING),
      models.embedding,
      "选择文本 Embedding",
    );
    fillSelect(
      document.getElementById("kc-visual-embedding"),
      modelsByCategory(KBotAssistantApi.ModelCategory.IMG_EMBEDDING),
      models.visual_embedding,
      "不绑定",
    );
  }

  function actionButtons(row) {
    const id = escapeHtml(row.collection_id);
    if (row.status === "DELETING") {
      return '<span class="assistant-badge warn">清理中</span>';
    }
    const buttons = [
      `<button class="small" type="button" data-action="documents" data-collection-id="${id}">资料</button>`,
      `<button class="small" type="button" data-action="edit" data-collection-id="${id}">编辑</button>`,
    ];
    if (row.status === "ACTIVE") {
      buttons.push(`<button class="small" type="button" data-action="disable" data-collection-id="${id}">停用</button>`);
    } else if (row.status === "DISABLED") {
      buttons.push(`<button class="small" type="button" data-action="enable" data-collection-id="${id}">启用</button>`);
    }
    buttons.push(`<button class="small danger" type="button" data-action="delete" data-collection-id="${id}">删除</button>`);
    return `<div class="assistant-row-actions">${buttons.join("")}</div>`;
  }

  function renderRows() {
    const node = document.getElementById("kc-rows");
    if (!node) return;
    if (!rows.length) {
      node.innerHTML = emptyRow(7, "当前 Domain 尚无 Knowledge Core", "需先创建或切换到有权限的 Domain。没有 ACTIVE Knowledge Core 的 Agent 只能保存草稿，不能启用。");
      return;
    }
    node.innerHTML = rows.map((row) => {
      const models = modelsOf(row);
      return `<tr>
        <td>${escapeHtml(row.display_name || "未命名")}</td>
        <td>${badge(row.status || "UNKNOWN", statusTone(row.status))}</td>
        <td>${escapeHtml(row.collection_id || "—")}</td>
        <td>${escapeHtml(modelLabel(models.embedding))}</td>
        <td>${escapeHtml(row.default_security_level ?? "—")}</td>
        <td>${escapeHtml(row.updated_at || "—")}</td>
        <td>${actionButtons(row)}</td>
      </tr>`;
    }).join("");
    node.querySelectorAll("[data-action]").forEach((button) => {
      button.addEventListener("click", () => {
        handleAction(button.dataset.action, button.dataset.collectionId).catch((error) => {
          toast(error.message || "Knowledge Core 操作失败", "error");
        });
      });
    });
  }

  function findRow(collectionId) {
    return rows.find((row) => String(row.collection_id) === String(collectionId));
  }

  async function loadPage() {
    const [listed, catalogRows] = await Promise.all([
      KBotAssistantApi.json("/knowledge-cores", "GET"),
      KBotAssistantApi.json("/model-catalog", "GET"),
    ]);
    rows = collectionItems(listed);
    catalog = KBotAssistantApi.items(catalogRows).filter((row) => KBotAssistantApi.isActiveModel(row));
    if (!selectedCollectionId || !findRow(selectedCollectionId)) {
      selectedCollectionId = rows[0]?.collection_id || null;
    }
    renderRows();
    await loadProcessing();
  }

  async function changeStatus(row, status) {
    await KBotAssistantApi.json(`/knowledge-cores/${row.collection_id}`, "PATCH", {
      expected_row_version: row.row_version,
      status,
    });
    await loadPage();
    toast(status === "ACTIVE" ? "Knowledge Core 已启用。" : "Knowledge Core 已停用。");
  }

  async function handleAction(action, collectionId) {
    const row = findRow(collectionId);
    if (!row) return;
    if (action === "edit") {
      fillForm(row);
      openDialog("kc-dialog");
      return;
    }
    if (action === "documents") {
      selectedCollectionId = collectionId;
      await loadProcessing();
      document.getElementById("kc-upload-open")?.focus();
      return;
    }
    if (action === "enable") {
      await changeStatus(row, "ACTIVE");
      return;
    }
    if (action === "disable") {
      await changeStatus(row, "DISABLED");
      return;
    }
    if (action === "delete") {
      const confirmed = globalThis.confirm(`删除会进入清理流程，不能撤销。确认删除「${row.display_name}」？`);
      if (!confirmed) return;
      await KBotAssistantApi.json(`/knowledge-cores/${row.collection_id}`, "DELETE");
      await loadPage();
      toast("已提交删除，Knowledge Core 正在清理。");
    }
  }

  function createPayload() {
    const embedding = document.getElementById("kc-embedding").value;
    const visual = document.getElementById("kc-visual-embedding").value;
    const payload = {
      display_name: String(document.getElementById("kc-name").value || "").trim(),
      description: String(document.getElementById("kc-description").value || "").trim() || null,
      default_security_level: Number(document.getElementById("kc-security").value || 1),
      embedding,
    };
    if (visual) payload.visual_embedding = visual;
    return payload;
  }

  async function submitForm(event) {
    event.preventDefault();
    const payload = createPayload();
    if (!payload.display_name) {
      toast("请填写 Knowledge Core 名称。", "error");
      return;
    }
    if (!payload.embedding) {
      toast("请选择已启用的文本 Embedding 模型。", "error");
      return;
    }
    try {
      if (editingId == null) {
        await KBotAssistantApi.json("/knowledge-cores", "POST", payload);
        closeDialog("kc-dialog");
        resetForm();
        await loadPage();
        toast("Knowledge Core 已创建。");
        return;
      }
      const current = findRow(editingId) || {};
      const models = modelsOf(current);
      const update = {
        expected_row_version: Number(document.getElementById("kc-row-version").value),
        display_name: payload.display_name,
        description: payload.description,
        default_security_level: payload.default_security_level,
      };
      if (payload.embedding !== String(models.embedding || "")) update.embedding = payload.embedding;
      const currentVisual = models.visual_embedding ? String(models.visual_embedding) : "";
      const nextVisual = payload.visual_embedding ? String(payload.visual_embedding) : "";
      if (nextVisual !== currentVisual) update.visual_embedding = payload.visual_embedding || null;
      await KBotAssistantApi.json(`/knowledge-cores/${editingId}`, "PATCH", update);
      closeDialog("kc-dialog");
      resetForm();
      await loadPage();
      toast("Knowledge Core 已保存。");
    } catch (error) {
      toast(error.message || "无法保存 Knowledge Core", "error");
    }
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

  async function uploadFiles(event) {
    event.preventDefault();
    if (!selectedCollectionId) {
      toast("请先选择一个 Knowledge Core。", "error");
      return;
    }
    const input = document.getElementById("kc-file-input");
    const files = Array.from(input?.files || []);
    if (!files.length) {
      toast("请选择至少一个文件。", "error");
      return;
    }
    const form = new FormData();
    const declarations = [];
    for (const [index, file] of files.entries()) {
      const partName = `file_${index}`;
      form.append(partName, file, file.name);
      declarations.push({
        part_name: partName,
        client_file_id: `${Date.now()}-${index}-${file.name}`,
        display_name: file.name,
        declared_mime_type: file.type || "application/octet-stream",
        byte_size: file.size,
        content_sha256: await sha256(file),
        ordinal: index,
        role: "CONTENT",
        required_flag: true,
      });
    }
    form.append("grouping_mode", "EACH_FILE");
    form.append("files", JSON.stringify(declarations));
    const button = document.getElementById("kc-upload-submit");
    if (button) button.disabled = true;
    try {
      await KBotAssistantApi.request(`/knowledge-cores/${selectedCollectionId}/ingestions/user-files`, {
        method: "POST",
        body: form,
        headers: { "Idempotency-Key": KBotAssistantApi.requestId() },
      });
      closeDialog("kc-upload-dialog");
      input.value = "";
      await loadProcessing();
      toast(`已受理 ${files.length} 个文件，正在处理。`);
    } finally {
      if (button) button.disabled = false;
    }
  }

  KBotAssistantShell.ready.then(async (access) => {
    if (!access) return;
    document.querySelectorAll("[data-open-dialog]").forEach((button) => {
      button.addEventListener("click", () => {
        resetForm();
        openDialog(button.dataset.openDialog);
      });
    });
    document.querySelectorAll("[data-close-dialog]").forEach((button) => {
      button.addEventListener("click", () => closeDialog(button.dataset.closeDialog));
    });
    document.getElementById("kc-form")?.addEventListener("submit", submitForm);
    document.getElementById("kc-upload-form")?.addEventListener("submit", uploadFiles);
    document.getElementById("kc-upload-open")?.addEventListener("click", () => openDialog("kc-upload-dialog"));
    document.getElementById("kc-refresh-processing")?.addEventListener("click", () => {
      loadProcessing().catch((error) => toast(error.message || "无法刷新解析进度", "error"));
    });
    try {
      await loadPage();
    } catch (error) {
      toast(error.message || "无法加载 Knowledge Core", "error");
    }
  });
})();
