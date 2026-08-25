(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  let collection = null;
  let policy = null;
  let models = [];

  const modelId = (value) => String(value || "");
  const selectedModel = (role) => modelId(collection?.models?.[role]);
  const modelLabel = (id) => {
    const row = models.find((item) => modelId(item.model_id) === modelId(id));
    return row ? `${row.display_name} · ${row.served_model_name}` : id || "未配置";
  };

  function categoryOptions(category, selected, allowEmpty) {
    const empty = allowEmpty ? '<option value="">不启用</option>' : "";
    return empty + models
      .filter((item) => Number(item.category) === category)
      .map((item) => `<option value="${shell.escape(item.model_id)}" ${modelId(item.model_id) === modelId(selected) ? "selected" : ""}>${shell.escape(item.display_name)} · ${shell.escape(item.served_model_name)}</option>`)
      .join("");
  }

  function render() {
    document.getElementById("collection-status").outerHTML = shell.badge(collection.status).replace("<span", '<span id="collection-status"');
    document.getElementById("collection-summary").innerHTML = [
      ["名称", collection.display_name],
      ["Collection ID", collection.collection_id],
      ["行版本", collection.row_version],
      ["内容类型", collection.metadata?.content_kind || "operations_manual"],
    ].map(([label, value]) => `<div class="kc-summary-item"><small>${label}</small><strong>${shell.escape(value ?? "—")}</strong></div>`).join("");

    const roles = [
      ["parser_vlm", "解析 VLM"],
      ["embedding", "文本向量模型"],
      ["visual_embedding", "视觉向量模型"],
    ];
    document.getElementById("model-list").innerHTML = roles.map(([role, label]) => `<div class="kc-model-row"><strong>${label}</strong><span>${shell.escape(modelLabel(selectedModel(role)))}</span>${shell.badge(selectedModel(role) ? "ACTIVE" : "NOT_CONFIGURED")}</div>`).join("");

    const embeddingLocked = policy?.embedding_change_allowed === false;
    const visualLocked = policy?.visual_embedding_change_allowed === false;
    const notes = [];
    if (embeddingLocked) notes.push("Collection 已有索引内容，文本向量模型不可直接更换；请按 KC 重建索引流程处理。");
    if (visualLocked) notes.push("Collection 已有视觉索引内容，视觉向量模型不可直接更换。");
    if (!notes.length) notes.push("当前 KC 策略允许保存所选模型；保存时仍会再次校验模型类别与行版本。");
    document.getElementById("model-policy-note").hidden = false;
    document.getElementById("model-policy-note").textContent = notes.join(" ");
    document.getElementById("dialog-policy-note").textContent = notes.join(" ");
    document.getElementById("parser-vlm").innerHTML = categoryOptions(5, selectedModel("parser_vlm"), true);
    document.getElementById("embedding-model").innerHTML = categoryOptions(2, selectedModel("embedding"), false);
    document.getElementById("visual-embedding").innerHTML = categoryOptions(3, selectedModel("visual_embedding"), true);
    document.getElementById("embedding-model").disabled = embeddingLocked;
    document.getElementById("visual-embedding").disabled = visualLocked;
  }

  async function load() {
    const result = await Promise.all([
      KBotAIOpsAuth.request(`${api}/model-catalog`),
      KBotAIOpsAuth.request(`${api}/knowledge-core`),
    ]);
    models = result[0];
    collection = result[1].collection;
    policy = result[1].model_policy;
    render();
  }

  async function saveModels(event) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const payload = {
      parser_vlm: form.get("parser_vlm") || null,
      embedding: form.get("embedding") || selectedModel("embedding"),
      visual_embedding: document.getElementById("visual-embedding").disabled
        ? (selectedModel("visual_embedding") || null)
        : (form.get("visual_embedding") || null),
      expected_row_version: Number(collection.row_version),
    };
    if (!payload.embedding) throw new Error("必须选择文本向量模型");
    await KBotAIOpsAuth.request(`${api}/knowledge-core/models`, {
      method: "PUT", body: JSON.stringify(payload),
    });
    document.getElementById("model-dialog").close();
    shell.toast("KC 模型配置已保存");
    await load();
  }

  async function uploadManual(event) {
    event.preventDefault();
    const button = event.currentTarget.querySelector('button[type="submit"]');
    const resultNode = document.getElementById("manual-result");
    const file = document.getElementById("manual-file").files[0];
    if (!file) return;
    button.disabled = true;
    resultNode.textContent = "正在计算摘要并上传…";
    try {
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", await file.arrayBuffer())))
        .map((value) => value.toString(16).padStart(2, "0")).join("");
      const partName = "manual";
      const declaration = [{
        part_name: partName,
        client_file_id: KBotAIOpsAuth.uuid(),
        display_name: document.getElementById("manual-name").value.trim() || file.name,
        declared_mime_type: file.type || "application/octet-stream",
        byte_size: file.size,
        content_sha256: digest,
        ordinal: 0,
        role: "CONTENT",
        required_flag: true,
      }];
      const form = new FormData();
      form.append("grouping_mode", "EACH_FILE");
      form.append("files", JSON.stringify(declaration));
      form.append(partName, file, file.name);
      const intake = await KBotAIOpsAuth.request(`${api}/knowledge-core/manuals`, {
        method: "POST",
        headers: { "Idempotency-Key": `aiops-manual:${digest}` },
        body: form,
      });
      const item = (intake.items || [])[0];
      if (!item?.bundle_revision_id || item.status === "REJECTED") {
        throw new Error(item?.message || "KC 未受理该运维手册");
      }
      await KBotAIOpsAuth.request(`${api}/knowledge-core/manuals/${encodeURIComponent(item.bundle_revision_id)}/approve`, {
        method: "POST",
        body: JSON.stringify({ comment: "AIOps 管理页面批准运维手册" }),
      });
      event.currentTarget.reset();
      resultNode.textContent = `上传并批准完成：${declaration[0].display_name}`;
      shell.toast("运维手册已进入 KC 索引队列");
    } catch (error) {
      resultNode.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  }

  shell.ready.then(async () => {
    const dialog = document.getElementById("model-dialog");
    document.getElementById("open-model-dialog").onclick = () => dialog.showModal();
    document.getElementById("close-model-dialog").onclick = () => dialog.close();
    document.getElementById("cancel-model-dialog").onclick = () => dialog.close();
    document.getElementById("model-form").addEventListener("submit", (event) => {
      saveModels(event).catch((error) => shell.toast(error.message));
    });
    document.getElementById("manual-form").addEventListener("submit", uploadManual);
    try {
      await load();
    } catch (error) {
      document.getElementById("collection-summary").innerHTML = `<div class="ops-error">${shell.escape(error.message)}</div>`;
      document.getElementById("open-model-dialog").disabled = true;
    }
  });
})();
