/* 模型配置管理：只经 Main API 操作，Secret 写入后不回显。 */
(function () {
  "use strict";

  const { badge, escapeHtml, toast } = KBotMediaShell;
  const categoryLabels = new Map([
    [1, "LLM"], [2, "文本向量"], [3, "视觉向量"], [5, "VLM"], [6, "OCR"],
  ]);
  let models = [];
  let providerOptions = [];
  let editingModel = null;

  function element(id) { return document.getElementById(id); }
  function categoryLabel(value) { return categoryLabels.get(Number(value)) || `类别 ${value}`; }
  function statusBadge(status) {
    const labels = { ACTIVE: "已启用", DRAFT: "草稿", ARCHIVED: "已归档" };
    const tone = status === "ACTIVE" ? "good" : status === "ARCHIVED" ? "bad" : "warn";
    return badge(labels[status] || status, tone);
  }
  function emptyRow(title, copy) {
    return `<tr><td colspan="7"><div class="media-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div></td></tr>`;
  }
  function currentModel(modelId) {
    return models.find((row) => String(row.model_id) === String(modelId));
  }
  function endpointText(row) { return row.api_endpoint || "本地模型 / 未配置"; }

  function updateSummary() {
    element("model-total").textContent = String(models.length);
    element("model-active").textContent = String(models.filter((row) => row.status === "ACTIVE").length);
    element("model-draft").textContent = String(models.filter((row) => row.status === "DRAFT").length);
    element("model-archived").textContent = String(models.filter((row) => row.status === "ARCHIVED").length);
  }

  function filteredModels() {
    const keyword = String(element("model-keyword")?.value || "").trim().toLowerCase();
    const category = element("model-category-filter")?.value || "";
    const status = element("model-status-filter")?.value || "";
    return models.filter((row) => {
      if (category && Number(row.category) !== Number(category)) return false;
      if (status && row.status !== status) return false;
      if (!keyword) return true;
      return [
        row.display_name, row.served_model_name, row.provider_model_name,
        row.provider, row.api_endpoint,
      ].some((value) => String(value || "").toLowerCase().includes(keyword));
    });
  }

  function actionButtons(row) {
    const buttons = [];
    if (row.status !== "ARCHIVED") {
      buttons.push(`<button class="small" type="button" data-model-action="edit" data-model-id="${escapeHtml(row.model_id)}">编辑</button>`);
      buttons.push(`<button class="small" type="button" data-model-action="references" data-model-id="${escapeHtml(row.model_id)}">引用</button>`);
      buttons.push(`<button class="small" type="button" data-model-action="status" data-model-id="${escapeHtml(row.model_id)}">${row.status === "ACTIVE" ? "停用" : "启用"}</button>`);
      buttons.push(`<button class="small danger" type="button" data-model-action="archive" data-model-id="${escapeHtml(row.model_id)}">归档</button>`);
    } else {
      buttons.push(`<button class="small" type="button" data-model-action="references" data-model-id="${escapeHtml(row.model_id)}">引用</button>`);
      buttons.push(`<button class="small danger" type="button" data-model-action="delete" data-model-id="${escapeHtml(row.model_id)}">删除</button>`);
    }
    return buttons.join("");
  }

  function renderModels() {
    const rows = filteredModels();
    const target = element("model-rows");
    if (!target) return;
    if (!rows.length) {
      target.innerHTML = emptyRow("没有匹配的模型配置", "调整筛选条件，或新增一个由 Provider 合同校验的模型配置。");
      return;
    }
    target.innerHTML = rows.map((row) => `<tr>
      <td><strong>${escapeHtml(row.display_name)}</strong><small class="model-secondary">${escapeHtml(row.served_model_name)} · ${escapeHtml(row.provider_model_name)}</small></td>
      <td>${escapeHtml(categoryLabel(row.category))}<small class="model-secondary">${escapeHtml(row.provider)}</small></td>
      <td><code class="model-endpoint" title="${escapeHtml(endpointText(row))}">${escapeHtml(endpointText(row))}</code></td>
      <td>${statusBadge(row.status)}</td>
      <td>${row.supports_image_generation ? badge("图片生成已验收", "good") : badge("未验收图片生成", "warn")}</td>
      <td>v${escapeHtml(row.row_version)}</td>
      <td><div class="model-row-actions">${actionButtons(row)}</div></td>
    </tr>`).join("");
    target.querySelectorAll("[data-model-action]").forEach((button) => {
      button.addEventListener("click", () => handleAction(button.dataset.modelAction, button.dataset.modelId));
    });
  }

  async function loadModels() {
    const [modelRows, optionRows] = await Promise.all([
      KBotMediaApi.json("/models", "GET"),
      KBotMediaApi.json("/models/provider-options", "GET"),
    ]);
    models = KBotMediaApi.items(modelRows);
    providerOptions = KBotMediaApi.items(optionRows);
    updateSummary();
    renderModels();
  }

  function optionsForCategory() {
    const category = Number(element("model-category").value);
    return providerOptions.filter((item) => Number(item.category) === category);
  }

  function selectedProviderOption() {
    const category = Number(element("model-category").value);
    const provider = element("model-provider").value;
    return providerOptions.find((item) => Number(item.category) === category && item.provider === provider);
  }

  function renderProviderContract() {
    const option = selectedProviderOption();
    const contract = element("model-provider-contract");
    const ociField = element("model-oci-secret-field");
    const secretFields = option?.secret_fields || [];
    const allowed = option?.allowed_model_params || [];
    const required = option?.required_fields || [];
    contract.textContent = option
      ? `必要字段：${required.join("、") || "无"}；允许参数：${allowed.join("、") || "无"}`
      : "当前类别没有可用 Provider。";
    ociField.hidden = !secretFields.includes("model_params.config_file");
  }

  function syncProviderOptions(selected) {
    const select = element("model-provider");
    const rows = optionsForCategory();
    select.innerHTML = rows.map((item) => `<option value="${escapeHtml(item.provider)}">${escapeHtml(item.provider)}</option>`).join("");
    if (selected && rows.some((item) => item.provider === selected)) select.value = selected;
    renderProviderContract();
  }

  function resetForm() {
    editingModel = null;
    element("model-form").reset();
    element("model-id").value = "";
    element("model-row-version").value = "";
    element("model-category").value = "1";
    element("model-status").value = "DRAFT";
    element("model-params").value = "{}";
    element("model-oci-config").value = "";
    element("model-category").disabled = false;
    element("model-provider").disabled = false;
    element("model-served-name").disabled = false;
    element("model-provider-name").disabled = false;
    element("model-status").disabled = false;
    syncProviderOptions();
  }

  function openCreateDialog() {
    resetForm();
    element("model-dialog-title").textContent = "新增模型";
    element("model-dialog-subtitle").textContent = "新配置由类别对应的模型服务持久化，建议先以草稿保存。";
    element("model-dialog").showModal();
  }

  function openEditDialog(row) {
    resetForm();
    editingModel = row;
    element("model-id").value = row.model_id;
    element("model-row-version").value = row.row_version;
    element("model-category").value = String(row.category);
    syncProviderOptions(row.provider);
    element("model-provider").value = row.provider;
    element("model-served-name").value = row.served_model_name;
    element("model-provider-name").value = row.provider_model_name;
    element("model-display-name").value = row.display_name;
    element("model-status").value = row.status;
    element("model-api-endpoint").value = row.api_endpoint || "";
    element("model-api-key").value = "";
    element("model-params").value = JSON.stringify(row.model_params || {}, null, 2);
    element("model-description").value = row.description || "";
    element("model-category").disabled = true;
    element("model-provider").disabled = true;
    element("model-served-name").disabled = true;
    element("model-provider-name").disabled = true;
    element("model-status").disabled = true;
    renderProviderContract();
    element("model-dialog-title").textContent = `编辑 ${row.display_name}`;
    element("model-dialog-subtitle").textContent = "服务名、上游模型标识、类别和 Provider 不可原地修改；Secret 留空即保留。";
    element("model-dialog").showModal();
  }

  function parseObject(text, fieldName, allowEmpty = false) {
    const raw = String(text || "").trim();
    if (!raw && allowEmpty) return null;
    let value;
    try { value = JSON.parse(raw || "{}"); }
    catch (_) { throw new Error(`${fieldName}必须是有效 JSON`); }
    if (!value || Array.isArray(value) || typeof value !== "object") {
      throw new Error(`${fieldName}必须是 JSON Object`);
    }
    return value;
  }

  function normalizedJson(value) {
    if (Array.isArray(value)) return value.map(normalizedJson);
    if (value && typeof value === "object") {
      return Object.fromEntries(
        Object.keys(value).sort().map((key) => [key, normalizedJson(value[key])]),
      );
    }
    return value;
  }

  function objectsEqual(left, right) {
    return JSON.stringify(normalizedJson(left)) === JSON.stringify(normalizedJson(right));
  }

  function validateEndpoint(endpoint, provider) {
    if (!endpoint || provider !== "oci") return;
    let parsed;
    try { parsed = new URL(endpoint); }
    catch (_) { throw new Error("OCI API Endpoint 必须是有效的 HTTP(S) 服务 Origin"); }
    if (!["http:", "https:"].includes(parsed.protocol)
      || parsed.username || parsed.password
      || (parsed.pathname && parsed.pathname !== "/")
      || parsed.search || parsed.hash) {
      throw new Error("OCI API Endpoint 只能填写服务 Origin，不能包含 API 路径、查询参数或片段");
    }
  }

  function formPayload() {
    const modelId = element("model-id").value;
    const option = selectedProviderOption();
    const modelParams = parseObject(element("model-params").value, "模型参数");
    const ociConfig = parseObject(element("model-oci-config").value, "OCI config_file", true);
    if (ociConfig) modelParams.config_file = ociConfig;
    const endpoint = element("model-api-endpoint").value.trim();
    validateEndpoint(endpoint, option?.provider);
    const apiKey = element("model-api-key").value.trim();
    const description = element("model-description").value.trim();
    if (!modelId) {
      const required = new Set(option?.required_fields || []);
      if (required.has("api_endpoint") && !endpoint) throw new Error("当前 Provider 要求填写 API Endpoint");
      if (required.has("api_key") && !apiKey) throw new Error("当前 Provider 要求填写 API Key / Token");
      if (required.has("model_params.config_file") && !ociConfig) throw new Error("新增 OCI 模型必须填写 config_file");
      return {
        served_model_name: element("model-served-name").value.trim(),
        display_name: element("model-display-name").value.trim(),
        provider_model_name: element("model-provider-name").value.trim(),
        category: Number(element("model-category").value),
        provider: element("model-provider").value,
        api_endpoint: endpoint || null,
        api_key: apiKey || null,
        status: element("model-status").value,
        model_params: modelParams,
        description: description || null,
      };
    }
    const payload = { expected_row_version: Number(element("model-row-version").value) };
    const displayName = element("model-display-name").value.trim();
    const normalizedEndpoint = endpoint || null;
    const normalizedDescription = description || null;
    if (displayName !== editingModel?.display_name) payload.display_name = displayName;
    if (normalizedEndpoint !== (editingModel?.api_endpoint || null)) payload.api_endpoint = normalizedEndpoint;
    if (normalizedDescription !== (editingModel?.description || null)) payload.description = normalizedDescription;
    if (ociConfig || !objectsEqual(modelParams, editingModel?.model_params || {})) {
      payload.model_params = modelParams;
    }
    if (apiKey) payload.api_key = apiKey;
    if (Object.keys(payload).length === 1) throw new Error("模型配置没有发生变化");
    return payload;
  }

  async function saveModel(event) {
    event.preventDefault();
    const submit = element("model-submit");
    submit.disabled = true;
    try {
      const modelId = element("model-id").value;
      const payload = formPayload();
      if (modelId) {
        const category = Number(element("model-category").value);
        await KBotMediaApi.json(`/models/${category}/${modelId}`, "PATCH", payload);
      } else {
        await KBotMediaApi.json("/models", "POST", payload);
      }
      element("model-dialog").close();
      await loadModels();
      toast(modelId ? "模型配置已更新。" : "模型配置已创建。");
    } catch (error) {
      toast(error.message || "保存模型配置失败", "error");
    } finally {
      submit.disabled = false;
    }
  }

  async function showReferences(row) {
    const dialog = element("model-references-dialog");
    const content = element("model-reference-content");
    content.innerHTML = "<p>正在检查所有业务引用…</p>";
    dialog.showModal();
    try {
      const result = await KBotMediaApi.json(`/models/${row.category}/${row.model_id}/references`, "GET");
      const references = result.references || [];
      const unavailable = result.unavailable_services || [];
      const notices = unavailable.length
        ? `<div class="media-notice warn">无法完成引用检查：${escapeHtml(unavailable.join("、"))}</div>` : "";
      const list = references.length
        ? `<ul class="model-reference-list">${references.map((item) => `<li><strong>${escapeHtml(item.service)} · ${escapeHtml(item.resource_type)}</strong><span>${escapeHtml(item.resource_id)} · ${escapeHtml(item.usage)}</span></li>`).join("")}</ul>`
        : '<div class="media-empty compact"><div><strong>当前没有业务引用</strong><p>可在确认后归档；只有已归档且未运行的模型才能删除。</p></div></div>';
      content.innerHTML = `${notices}${list}`;
    } catch (error) {
      content.innerHTML = `<div class="media-notice warn">${escapeHtml(error.message || "引用检查失败")}</div>`;
    }
  }

  async function changeStatus(row) {
    const next = row.status === "ACTIVE" ? "DRAFT" : "ACTIVE";
    const verb = next === "ACTIVE" ? "启用" : "停用";
    if (!confirm(`确认${verb}模型“${row.display_name}”？`)) return;
    await KBotMediaApi.json(`/models/${row.category}/${row.model_id}/status`, "PATCH", {
      expected_row_version: row.row_version,
      status: next,
    });
    await loadModels();
    toast(`模型已${verb}。`);
  }

  async function archiveModel(row) {
    if (!confirm(`确认归档模型“${row.display_name}”？归档不可逆。`)) return;
    await KBotMediaApi.json(`/models/${row.category}/${row.model_id}/archive`, "POST", {
      expected_row_version: row.row_version,
    });
    await loadModels();
    toast("模型已归档。");
  }

  async function deleteModel(row) {
    if (!confirm(`确认永久删除已归档模型“${row.display_name}”？此操作不可恢复。`)) return;
    await KBotMediaApi.json(`/models/${row.category}/${row.model_id}`, "DELETE", {
      expected_row_version: row.row_version,
    });
    await loadModels();
    toast("已归档模型已删除。");
  }

  async function handleAction(action, modelId) {
    const row = currentModel(modelId);
    if (!row) return;
    try {
      if (action === "edit") return openEditDialog(row);
      if (action === "references") return await showReferences(row);
      if (action === "status") return await changeStatus(row);
      if (action === "archive") return await archiveModel(row);
      if (action === "delete") return await deleteModel(row);
    } catch (error) {
      toast(error.message || "模型操作失败", "error");
    }
  }

  KBotMediaShell.ready.then(async (access) => {
    if (!access || document.body.dataset.page !== "models") return;
    element("model-create")?.addEventListener("click", openCreateDialog);
    element("models-refresh")?.addEventListener("click", () => loadModels().catch((error) => toast(error.message, "error")));
    element("model-form")?.addEventListener("submit", saveModel);
    element("model-category")?.addEventListener("change", () => syncProviderOptions());
    element("model-provider")?.addEventListener("change", renderProviderContract);
    element("model-keyword")?.addEventListener("input", renderModels);
    element("model-category-filter")?.addEventListener("change", renderModels);
    element("model-status-filter")?.addEventListener("change", renderModels);
    document.querySelectorAll("[data-close-model-dialog]").forEach((button) => button.addEventListener("click", () => element("model-dialog").close()));
    document.querySelectorAll("[data-close-reference-dialog]").forEach((button) => button.addEventListener("click", () => element("model-references-dialog").close()));
    try { await loadModels(); }
    catch (error) {
      element("model-rows").innerHTML = emptyRow("模型配置读取失败", error.message || "请检查模型服务状态。");
      toast(error.message || "无法读取模型配置", "error");
    }
  });
})();
