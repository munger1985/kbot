/* 模型配置管理：只经 Main API 操作，Secret 写入后不回显。 */
import { parameterLabel, renderModelParameterFields } from "./model-parameter-fields.js?v=20261122_local_embedding_metadata";

(function () {
  "use strict";

  const session = KBotPlatformSession;
  const portal = "/ui/index.html";
  const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
  const badge = (value, tone) => `<span class="media-badge ${escapeHtml(tone)}">${escapeHtml(value)}</span>`;
  const confirmAction = (options) => import("../shared/kbot-confirm.js").then((module) => module.confirmAction(options));

  function toast(message, kind = "info") {
    const item = document.createElement("div");
    item.className = `media-toast ${kind === "error" ? "error" : ""}`;
    item.textContent = message;
    element("system-toast").append(item);
    setTimeout(() => item.remove(), 4500);
  }

  async function modelRequest(path, method, body) {
    try {
      return await session.request(`/platform${path}`, method, body);
    } catch (error) {
      if (error.status === 401) location.replace(portal);
      if (error.code === "PLATFORM_PERMISSION_DENIED" || error.code === "PLATFORM_SESSION_REQUIRED") {
        element("system-content").hidden = true;
        for (const dialog of document.querySelectorAll("dialog[open]")) dialog.close();
        element("system-status").textContent = "当前账号没有管理 AI 模型的权限。";
        element("system-status").hidden = false;
      }
      throw error;
    }
  }
  const categoryLabels = new Map([
    [1, "LLM"], [2, "文本向量"], [3, "视觉向量"], [5, "VLM"], [6, "OCR"], [8, "图像生成"],
  ]);
  let models = [];
  let providerOptions = [];
  let editingModel = null;
  const testingModels = new Set();
  let parameterFields;
  let credentialVersion = 0;
  const ociAuthFieldIds = {
    tenancy: "model-oci-tenancy",
    user: "model-oci-user",
    fingerprint: "model-oci-fingerprint",
    region: "model-oci-region",
    key_content: "model-oci-key-content",
    pass_phrase: "model-oci-pass-phrase",
  };

  function element(id) { return document.getElementById(id); }
  function categoryLabel(value) { return categoryLabels.get(Number(value)) || `类别 ${value}`; }
  function statusBadge(status) {
    const labels = { ACTIVE: "已启用", DRAFT: "草稿", ARCHIVED: "已归档" };
    const tone = status === "ACTIVE" ? "good" : status === "ARCHIVED" ? "bad" : "warn";
    return badge(labels[status] || status, tone);
  }
  function emptyRow(title, copy) {
    return `<tr><td colspan="6"><div class="media-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div></td></tr>`;
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
        row.display_name, row.model_id, row.provider_model_name,
        row.provider, row.api_endpoint,
      ].some((value) => String(value || "").toLowerCase().includes(keyword));
    });
  }

  function actionButtons(row) {
    const buttons = [];
    if (row.status !== "ARCHIVED") {
      buttons.push(`<button class="small" type="button" data-model-action="edit" data-model-id="${escapeHtml(row.model_id)}">编辑</button>`);
      const testing = testingModels.has(String(row.model_id));
      const testHint = row.status === "ACTIVE" ? "用已保存的配置测试连通性" : "请先启用模型，再测试连通性";
      buttons.push(`<button class="small" type="button" data-model-action="test" data-model-id="${escapeHtml(row.model_id)}" title="${testHint}" ${testing || row.status !== "ACTIVE" ? "disabled" : ""}>${testing ? "测试中…" : "测试"}</button>`);
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
      <td><strong>${escapeHtml(row.display_name)}</strong>
        <small class="model-secondary">厂商模型：${escapeHtml(row.provider_model_name)}</small></td>
      <td>${escapeHtml(categoryLabel(row.category))}<small class="model-secondary">${escapeHtml(row.provider)}</small></td>
      <td><code class="model-endpoint" title="${escapeHtml(endpointText(row))}">${escapeHtml(endpointText(row))}</code></td>
      <td>${statusBadge(row.status)}</td>
      <td>v${escapeHtml(row.row_version)}</td>
      <td><div class="model-row-actions">${actionButtons(row)}</div></td>
    </tr>`).join("");
    target.querySelectorAll("[data-model-action]").forEach((button) => {
      button.addEventListener("click", () => handleAction(button.dataset.modelAction, button.dataset.modelId));
    });
  }

  async function loadModels() {
    const [modelRows, optionRows] = await Promise.all([
      modelRequest("/models", "GET"),
      modelRequest("/models/provider-options", "GET"),
    ]);
    models = modelRows;
    providerOptions = optionRows;
    element("model-create").disabled = !providerOptions.length;
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

  function usesOciAuth(option) {
    const fields = option?.secret_fields || [];
    return fields.includes("model_params.config_file")
      || fields.includes("model_params.oci_auth");
  }

  function usesApiKey(option) {
    return option?.required_fields?.includes("api_key")
      || (usesOciAuth(option) && option.secret_fields?.includes("api_key"));
  }

  function clearOciAuthFields() {
    credentialVersion += 1;
    Object.values(ociAuthFieldIds).forEach((id) => { element(id).value = ""; });
    element("model-oci-config-upload").value = "";
    element("model-oci-key-upload").value = "";
  }

  function applyOciAuthFields(values) {
    Object.entries(ociAuthFieldIds).forEach(([key, id]) => {
      if (values[key] != null && String(values[key]).trim()) {
        element(id).value = String(values[key]).trim();
      }
    });
  }

  function parseOciConfigText(text) {
    const raw = String(text || "").trim();
    if (!raw) throw new Error("OCI config 文件为空");
    if (raw.startsWith("{")) {
      const parsed = parseObject(raw, "OCI config 文件");
      return { profile: "JSON", values: parsed };
    }
    const profiles = new Map();
    let profile = "DEFAULT";
    profiles.set(profile, {});
    raw.split(/\r?\n/).forEach((sourceLine) => {
      const line = sourceLine.trim();
      if (!line || line.startsWith("#") || line.startsWith(";")) return;
      const section = line.match(/^\[([^\]]+)]$/);
      if (section) {
        profile = section[1].trim() || "DEFAULT";
        if (!profiles.has(profile)) profiles.set(profile, {});
        return;
      }
      const separator = line.indexOf("=");
      if (separator < 1) return;
      const key = line.slice(0, separator).trim();
      const value = line.slice(separator + 1).trim();
      profiles.get(profile)[key] = value;
    });
    const selectedProfile = profiles.has("DEFAULT") && Object.keys(profiles.get("DEFAULT")).length
      ? "DEFAULT"
      : Array.from(profiles.entries()).find(([, values]) => Object.keys(values).length)?.[0];
    const values = profiles.get(selectedProfile) || {};
    if (!Object.keys(values).length) throw new Error("OCI config 文件中没有可解析的 Profile");
    return { profile: selectedProfile, values };
  }

  async function loadOciConfigFile(file) {
    if (!file || !usesOciAuth(selectedProviderOption())) return;
    const version = credentialVersion;
    const contents = await file.text();
    if (version !== credentialVersion) return;
    const parsed = parseOciConfigText(contents);
    applyOciAuthFields(parsed.values);
    const keyHint = parsed.values.key_file && !parsed.values.key_content
      ? "；本机 key_file 路径不会上传，请继续选择 API 私钥文件" : "";
    toast(`已解析 OCI Profile：${parsed.profile}${keyHint}`);
  }

  async function loadOciPrivateKey(file) {
    if (!file || !usesOciAuth(selectedProviderOption())) return;
    const version = credentialVersion;
    const content = String(await file.text()).trim();
    if (version !== credentialVersion) return;
    if (!content.includes("PRIVATE KEY")) throw new Error("所选文件不是有效的 PEM 私钥");
    element("model-oci-key-content").value = content;
    toast("OCI API 私钥已读取，保存后不会回显。");
  }

  function buildOciAuth() {
    const values = Object.fromEntries(
      Object.entries(ociAuthFieldIds).map(([key, id]) => [key, element(id).value.trim()]),
    );
    const supplied = Object.values(values).some(Boolean);
    if (!supplied) return null;
    const required = ["tenancy", "user", "fingerprint", "region", "key_content"];
    const missing = required.filter((key) => !values[key]);
    if (missing.length) {
      throw new Error(`OCI 认证参数不完整：${missing.join("、")}`);
    }
    if (!values.pass_phrase) delete values.pass_phrase;
    return values;
  }

  function renderProviderContract() {
    const option = selectedProviderOption();
    const oci = usesOciAuth(option);
    const keyField = usesApiKey(option);
    const endpointField = option?.required_fields?.includes("api_endpoint");
    element("model-api-key-field").hidden = !keyField;
    element("model-api-key").disabled = !keyField;
    element("model-api-key-label").textContent = oci ? "OCI API Token（令牌模式）" : "API Key / Token";
    element("model-endpoint-field").hidden = !endpointField;
    element("model-api-endpoint").disabled = !endpointField;
    element("model-oci-endpoint-help").hidden = !oci;
    element("model-oci-secret-field").hidden = !oci;
    parameterFields = renderModelParameterFields(
      element("model-parameter-fields"), option, editingModel?.model_params, oci,
    );
    element("model-params").hidden = !parameterFields.count;
  }

  function providerChanged() {
    element("model-api-key").value = "";
    element("model-api-endpoint").value = "";
    clearOciAuthFields();
    renderProviderContract();
  }

  function syncProviderOptions(selected) {
    const select = element("model-provider");
    const rows = optionsForCategory();
    select.innerHTML = rows.map((item) => `<option value="${escapeHtml(item.provider)}">${escapeHtml(item.provider)}</option>`).join("");
    if (selected && rows.some((item) => item.provider === selected)) select.value = selected;
    providerChanged();
  }

  function resetForm() {
    editingModel = null;
    element("model-form").reset();
    element("model-id").value = "";
    element("model-row-version").value = "";
    element("model-category").value = "1";
    element("model-status").value = "DRAFT";
    element("model-category").disabled = false;
    element("model-provider").disabled = false;
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
    element("model-provider-name").value = row.provider_model_name;
    element("model-display-name").value = row.display_name;
    element("model-status").value = row.status;
    element("model-api-endpoint").value = row.api_endpoint || "";
    element("model-api-key").value = "";
    element("model-description").value = row.description || "";
    element("model-category").disabled = true;
    element("model-provider").disabled = true;
    element("model-status").disabled = true;
    element("model-dialog-title").textContent = `编辑 ${row.display_name}`;
    element("model-dialog-subtitle").textContent = [2, 3].includes(Number(row.category))
      ? "可以更新厂商模型名称，但不要把已有索引切到不同的向量空间。"
      : "厂商模型名称可修改。凭据留空表示保留已有配置。";
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

  function validateEndpoint(endpoint, oci) {
    if (!endpoint || !oci) return;
    let parsed;
    try { parsed = new URL(endpoint); }
    catch (_) { throw new Error("OCI API Endpoint 必须是有效的 HTTP(S) 服务 Origin"); }
    if (!["http:", "https:"].includes(parsed.protocol)
      || parsed.username || parsed.password
      || parsed.search || parsed.hash) {
      throw new Error("OCI API Endpoint 必须是无用户信息、查询参数和片段的 HTTP(S) URL");
    }
  }

  function formPayload() {
    const modelId = element("model-id").value;
    const option = selectedProviderOption();
    if (!option) throw new Error("请选择有效的 Provider。");
    const oci = usesOciAuth(option);
    const modelParams = parameterFields.values();
    const ociAuth = oci ? buildOciAuth() : null;
    if (ociAuth) {
      delete modelParams.config_file;
      modelParams.oci_auth = ociAuth;
    }
    const endpointInput = element("model-api-endpoint");
    const endpoint = endpointInput.disabled ? null : endpointInput.value.trim();
    validateEndpoint(endpoint, oci);
    const apiKey = usesApiKey(option) ? element("model-api-key").value.trim() : "";
    const description = element("model-description").value.trim();
    const providerModelName = element("model-provider-name").value.trim();
    if (!providerModelName) throw new Error("请填写厂商模型名称或 ID。");
    if (!modelId) {
      const required = new Set(option?.required_fields || []);
      if (required.has("api_endpoint") && !endpoint) throw new Error("当前 Provider 要求填写 API 地址");
      if (required.has("api_key") && !apiKey) throw new Error("当前 Provider 要求填写 API Key / Token");
      for (const path of required) {
        if (!path.startsWith("model_params.")) continue;
        const key = path.slice("model_params.".length);
        if (oci && apiKey && ["compartment_id", "project"].includes(key)) continue;
        if (modelParams[key] == null || modelParams[key] === "") {
          throw new Error(`当前 Provider 要求填写：${parameterLabel(key, option.category)}`);
        }
      }
      if (oci && !apiKey && !ociAuth) {
        throw new Error(usesApiKey(option) ? "请填写 OCI API Token，或配置 OCI 签名认证。" : "请配置 OCI 签名认证。");
      }
      return {
        display_name: element("model-display-name").value.trim(),
        provider_model_name: providerModelName,
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
    if (providerModelName !== editingModel?.provider_model_name) payload.provider_model_name = providerModelName;
    if (!endpointInput.disabled && normalizedEndpoint !== (editingModel?.api_endpoint || null)) payload.api_endpoint = normalizedEndpoint;
    if (normalizedDescription !== (editingModel?.description || null)) payload.description = normalizedDescription;
    if (ociAuth || !objectsEqual(modelParams, editingModel?.model_params || {})) {
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
      let saved;
      if (modelId) {
        const category = Number(element("model-category").value);
        saved = await modelRequest(`/models/${category}/${modelId}`, "PATCH", payload);
      } else {
        saved = await modelRequest("/models", "POST", payload);
        if (saved?.status !== payload.status) {
          saved = await modelRequest(
            `/models/${saved.category}/${saved.model_id}/status`,
            "PATCH",
            {
              expected_row_version: saved.row_version,
              status: payload.status,
            },
          );
        }
      }
      element("model-dialog").close();
      await loadModels();
      const finalStatus = saved?.status === "ACTIVE" ? "已启用" : "草稿";
      toast(modelId ? "模型配置已更新。" : `模型配置已创建，当前状态：${finalStatus}。`);
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
      const result = await modelRequest(`/models/${row.category}/${row.model_id}/references`, "GET");
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

  async function testConnection(row) {
    const modelId = String(row.model_id);
    if (row.status !== "ACTIVE" || testingModels.has(modelId)) return;
    if (Number(row.category) === 8 && !await confirmAction({
      title: "测试图像生成模型？",
      message: "将向厂商发送一次真实请求，生成一张测试图，可能消耗额度或产生费用。测试图片不会保存。",
      confirmLabel: "开始测试",
      tone: "warning",
    })) return;
    testingModels.add(modelId);
    renderModels();
    try {
      const result = await modelRequest(`/models/${row.category}/${modelId}:test-connection`, "POST");
      if (!Number.isInteger(result?.latency_ms) || result.latency_ms < 0) throw new Error("模型服务未返回有效的测试结果");
      toast(`模型「${row.display_name}」测试通过，耗时 ${result.latency_ms} ms。`);
    } finally {
      testingModels.delete(modelId);
      renderModels();
    }
  }

  async function changeStatus(row) {
    const next = row.status === "ACTIVE" ? "DRAFT" : "ACTIVE";
    const verb = next === "ACTIVE" ? "启用" : "停用";
    if (!await confirmAction({ title: "确认模型操作？", message: `确认${verb}模型“${row.display_name}”？`, confirmLabel: "确认", tone: "danger" })) return;
    await modelRequest(`/models/${row.category}/${row.model_id}/status`, "PATCH", {
      expected_row_version: row.row_version,
      status: next,
    });
    await loadModels();
    toast(`模型已${verb}。`);
  }

  async function archiveModel(row) {
    if (!await confirmAction({ title: "确认模型操作？", message: `确认归档模型“${row.display_name}”？归档不可逆。`, confirmLabel: "确认", tone: "danger" })) return;
    await modelRequest(`/models/${row.category}/${row.model_id}/archive`, "POST", {
      expected_row_version: row.row_version,
    });
    await loadModels();
    toast("模型已归档。");
  }

  async function deleteModel(row) {
    if (!await confirmAction({ title: "确认模型操作？", message: `确认永久删除已归档模型“${row.display_name}”？此操作不可恢复。`, confirmLabel: "确认", tone: "danger" })) return;
    await modelRequest(`/models/${row.category}/${row.model_id}`, "DELETE", {
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
      if (action === "test") return await testConnection(row);
      if (action === "references") return await showReferences(row);
      if (action === "status") return await changeStatus(row);
      if (action === "archive") return await archiveModel(row);
      if (action === "delete") return await deleteModel(row);
    } catch (error) {
      toast(error.message || "模型操作失败", "error");
    }
  }

  async function initialize() {
    const user = session.load();
    if (!user || user.must_change_password) { location.replace(portal); return; }
    element("system-user").textContent = user.display_name || user.user_id;
    try {
      const entries = await session.request("/auth/entries");
      if (entries.system_management !== true) {
        element("system-status").textContent = "当前账号没有管理 AI 模型的权限。";
        return;
      }
    } catch (error) {
      if (error.status === 401) location.replace(portal);
      else element("system-status").textContent = error.message || "无法读取系统管理权限，请刷新重试。";
      return;
    }
    element("system-content").hidden = false;
    element("system-status").hidden = true;
    element("model-create")?.addEventListener("click", openCreateDialog);
    element("models-refresh")?.addEventListener("click", () => loadModels().catch((error) => toast(error.message, "error")));
    element("model-form").addEventListener("submit", saveModel);
    element("model-dialog").addEventListener("close", () => {
      element("model-api-key").value = "";
      clearOciAuthFields();
    });
    element("model-category").addEventListener("change", () => syncProviderOptions());
    element("model-provider").addEventListener("change", providerChanged);
    element("model-oci-config-upload")?.addEventListener("change", (event) => {
      loadOciConfigFile(event.target.files?.[0]).catch((error) => toast(error.message, "error"));
    });
    element("model-oci-key-upload")?.addEventListener("change", (event) => {
      loadOciPrivateKey(event.target.files?.[0]).catch((error) => toast(error.message, "error"));
    });
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
  }
  addEventListener("pageshow", (event) => { if (event.persisted) location.reload(); });
  initialize();
})();
