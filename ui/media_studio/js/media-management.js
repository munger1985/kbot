/* 模型绑定、图片资产与运行记录：只调用公开 Main API，预览使用 blob。 */
(function () {
  "use strict";

  const { badge, escapeHtml, toast } = KBotMediaShell;
  const previewUrls = [];
  let bindings = [];
  let catalog = [];

  function openDialog(id) { document.getElementById(id)?.showModal(); }
  function closeDialog(id) { document.getElementById(id)?.close(); }

  function emptyRow(columns, title, copy) {
    return `<tr><td colspan="${columns}"><div class="media-empty"><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(copy)}</p></div></div></td></tr>`;
  }

  function revokePreviews() {
    while (previewUrls.length) URL.revokeObjectURL(previewUrls.pop());
  }

  function roleLabel(role) { return role === "IMAGE_GENERATION" ? "图片生成" : role; }

  function roleRequirement(role) { return role === "IMAGE_GENERATION" ? "已启用的图像生成模型" : "已启用模型"; }

  function kindLabel(kind) { return kind === "IMAGE_GENERATION" ? "图片生成" : kind === "IMAGE_EDIT" ? "图片编辑" : kind; }

  function fillSelect(select, rows, currentId) {
    const options = [`<option value="">未选择</option>`].concat(
      rows.map((row) => {
        const selected = String(row.model_id) === String(currentId || "") ? " selected" : "";
        const active = KBotMediaApi.isActiveModel(row);
        const disabled = active ? "" : " disabled";
        const status = active ? "已启用" : row.status === "ARCHIVED" ? "已归档" : "草稿";
        const label = `${row.display_name || row.served_model_name || row.model_id}（${status}）`;
        return `<option value="${escapeHtml(row.model_id)}"${selected}${disabled}>${escapeHtml(label)}</option>`;
      }),
    );
    select.innerHTML = options.join("");
    select.disabled = false;
  }

  function currentBinding(role) {
    return bindings.find((row) => row.role === role);
  }

  async function openBindingEditor(role) {
    const current = currentBinding(role);
    const select = document.getElementById("image-model");
    select.innerHTML = '<option value="">正在读取最新图像模型目录…</option>';
    select.disabled = true;
    const title = document.getElementById("model-binding-dialog-title");
    if (title) title.textContent = current ? `编辑${roleLabel(role)}模型绑定` : "配置模型绑定";
    openDialog("model-binding-dialog");
    try {
      const catalogRows = await KBotMediaApi.json("/model-catalog", "GET");
      catalog = KBotMediaApi.items(catalogRows);
      const models = catalog.filter(
        (row) => Number(row.category) === KBotMediaApi.ModelCategory.IMAGE_GENERATION,
      );
      fillSelect(select, models, current?.model_id);
      if (!models.length) {
        select.innerHTML = '<option value="">暂无图像生成模型</option>';
      }
    } catch (error) {
      select.innerHTML = '<option value="">图像模型目录读取失败</option>';
      toast(error.message || "无法读取最新图像模型目录", "error");
    }
  }

  function renderBindings() {
    const node = document.getElementById("binding-rows");
    if (!node) return;
    const roles = ["IMAGE_GENERATION"];
    node.innerHTML = roles.map((role) => {
      const row = currentBinding(role);
      const status = row ? (row.ready ? badge("已绑定", "good") : badge("模型未启用", "warn")) : badge("未绑定", "warn");
      return `<tr>
        <td>${escapeHtml(roleLabel(role))}</td>
        <td>${escapeHtml(row?.display_name || "等待模型目录")}</td>
        <td>${escapeHtml(roleRequirement(role))}</td>
        <td>${status}</td>
        <td>${escapeHtml(row?.updated_at || "—")}</td>
        <td><button class="small" type="button" data-edit-binding="${escapeHtml(role)}">${row ? "编辑" : "配置"}</button></td>
      </tr>`;
    }).join("");
  }

  async function loadBindingsPage() {
    const [bindingRows, catalogRows] = await Promise.all([
      KBotMediaApi.json("/bindings", "GET"),
      KBotMediaApi.json("/model-catalog", "GET"),
    ]);
    bindings = Array.isArray(bindingRows) ? bindingRows : KBotMediaApi.items(bindingRows);
    catalog = KBotMediaApi.items(catalogRows);
    renderBindings();
  }

  async function saveBindings(event) {
    event.preventDefault();
    const entries = [["IMAGE_GENERATION", "image-model"]];
    let saved = 0;
    try {
      for (const [role, id] of entries) {
        const modelId = document.getElementById(id)?.value;
        if (!modelId) continue;
        const current = currentBinding(role);
        const payload = { role, model_id: modelId };
        if (current?.row_version) payload.expected_row_version = current.row_version;
        await KBotMediaApi.json("/bindings", "PUT", payload);
        saved += 1;
      }
      if (!saved) {
        toast("请选择一个已启用的图像生成模型后再保存。", "error");
        return;
      }
      await loadBindingsPage();
      closeDialog("model-binding-dialog");
      toast("模型绑定已保存。");
    } catch (error) {
      toast(error.message || "保存模型绑定失败", "error");
    }
  }

  function usageSummary(row) {
    const usage = row.usage || {};
    const parts = [];
    if (usage.input_tokens != null) parts.push(`入 ${usage.input_tokens}`);
    if (usage.output_tokens != null) parts.push(`出 ${usage.output_tokens}`);
    return parts.join(" / ") || "—";
  }

  function previewMeta(row) {
    const size = row.width && row.height ? `${row.width}×${row.height}` : "";
    return [row.asset_id, row.mime_type, size, row.created_at].filter(Boolean).join(" · ") || "受控预览";
  }

  async function previewAsset(row) {
    const dialog = document.getElementById("asset-preview-dialog");
    const body = document.getElementById("asset-preview-body");
    const title = document.getElementById("asset-preview-title");
    const meta = document.getElementById("asset-preview-meta");
    if (!dialog || !body) return;
    revokePreviews();
    if (title) title.textContent = "图片预览";
    if (meta) meta.textContent = previewMeta(row);
    body.innerHTML = `<p class="image-preview-meta">正在打开受控预览…</p>`;
    dialog.showModal();
    try {
      const blob = await KBotMediaApi.requestBlob(`/media-assets/${row.asset_id}/content`);
      const objectUrl = URL.createObjectURL(blob);
      previewUrls.push(objectUrl);
      body.innerHTML = `<img alt="图片资产 ${escapeHtml(row.asset_id)}" src="${objectUrl}">`;
    } catch (error) {
      body.innerHTML = `<p class="image-preview-meta">${escapeHtml(error.message || "无法预览图片资产")}</p>`;
      throw error;
    }
  }

  async function loadAssets() {
    const node = document.getElementById("asset-rows");
    if (!node) return;
    const status = document.getElementById("asset-status")?.value || "";
    const keyword = String(document.getElementById("asset-keyword")?.value || "").trim().toLowerCase();
    const period = document.getElementById("asset-period")?.value || "";
    const query = status ? KBotMediaApi.withQuery("/media-assets", { status }) : "/media-assets";
    const rows = await KBotMediaApi.json(query, "GET");
    const now = Date.now();
    const filtered = (Array.isArray(rows) ? rows : []).filter((row) => {
      if (keyword && !`${row.asset_id} ${row.run_id}`.toLowerCase().includes(keyword)) return false;
      if (period && row.created_at) {
        const created = Date.parse(row.created_at);
        const days = period === "today" ? 1 : period === "7d" ? 7 : period === "30d" ? 30 : 0;
        if (days && now - created > days * 86400000) return false;
      }
      return true;
    });
    if (!filtered.length) {
      node.innerHTML = emptyRow(7, "尚无可访问图片资产", "文生图模型通过验收并完成任务后，才会展示真实预览、尺寸、模型快照、提示词版本和受控下载操作。");
      return;
    }
    node.innerHTML = filtered.map((row) => `<tr>
      <td>${escapeHtml(row.asset_id)}</td>
      <td>${badge(row.status, row.status === "READY" ? "good" : "warn")}</td>
      <td>${escapeHtml([row.width && row.height ? `${row.width}×${row.height}` : "", row.mime_type].filter(Boolean).join(" / ") || "—")}</td>
      <td>${escapeHtml(row.provider_artifact_id || "已绑定模型")}</td>
      <td>${escapeHtml(row.prompt_revision_id || "—")}</td>
      <td>${escapeHtml(row.created_at || "—")}</td>
      <td><button class="small" type="button" data-preview-asset="${escapeHtml(row.asset_id)}">预览</button></td>
    </tr>`).join("");
    node.querySelectorAll("[data-preview-asset]").forEach((button) => {
      button.addEventListener("click", async () => {
        const assetId = button.dataset.previewAsset;
        const row = filtered.find((item) => String(item.asset_id) === String(assetId)) || { asset_id: assetId };
        try {
          await previewAsset(row);
        } catch (error) {
          toast(error.message || "无法预览图片资产", "error");
        }
      });
    });
  }

  async function loadRuns() {
    const node = document.getElementById("run-rows");
    if (!node) return;
    const kind = document.getElementById("run-type")?.value || "";
    const status = document.getElementById("run-status")?.value || "";
    const keyword = String(document.getElementById("run-keyword")?.value || "").trim().toLowerCase();
    const query = KBotMediaApi.withQuery("/runs", { kind, status });
    const rows = await KBotMediaApi.json(query, "GET");
    const filtered = (Array.isArray(rows) ? rows : []).filter((row) => {
      if (!keyword) return true;
      return `${row.run_id} ${row.request_id} ${row.provider_request_id || ""}`.toLowerCase().includes(keyword);
    });
    if (!filtered.length) {
      node.innerHTML = emptyRow(7, "尚无可显示的运行", "API 接入后，可在此恢复终态、打开引用或资产详情；不会将供应商原始账单、Secret 或完整 Prompt 直接作为列表字段返回。");
      return;
    }
    node.innerHTML = filtered.map((row) => `<tr>
      <td>${escapeHtml(row.run_id)}</td>
      <td>${escapeHtml(kindLabel(row.kind))}</td>
      <td>${badge(row.status, row.status === "COMPLETED" ? "good" : TERMINAL(row.status))}</td>
      <td>${escapeHtml(row.model_display_name || "已绑定模型")}</td>
      <td>${escapeHtml(usageSummary(row))}</td>
      <td>${escapeHtml(row.started_at || row.created_at || "—")}</td>
      <td>${escapeHtml(row.error_code || "查看详情")}</td>
    </tr>`).join("");
  }

  function TERMINAL(status) {
    return status === "FAILED" || status === "REJECTED" ? "bad" : "warn";
  }

  KBotMediaShell.ready.then(async (access) => {
    if (!access) return;
    document.querySelectorAll("[data-open-dialog]").forEach((button) => {
      button.addEventListener("click", () => openDialog(button.dataset.openDialog));
    });
    document.addEventListener("click", (event) => {
      const button = event.target.closest("[data-edit-binding]");
      if (button) {
        openBindingEditor(button.dataset.editBinding).catch(
          (error) => toast(error.message || "无法打开模型绑定配置", "error"),
        );
      }
    });
    document.querySelectorAll("[data-close-dialog]").forEach((button) => {
      button.addEventListener("click", () => closeDialog(button.dataset.closeDialog));
    });
    document.getElementById("asset-preview-dialog")?.addEventListener("close", revokePreviews);
    document.querySelector("#model-binding-dialog form")?.addEventListener("submit", saveBindings);
    document.getElementById("asset-filter")?.addEventListener("click", () => {
      loadAssets().catch((error) => toast(error.message || "无法加载图片资产", "error"));
    });
    document.getElementById("runs-filter")?.addEventListener("click", () => {
      loadRuns().catch((error) => toast(error.message || "无法加载运行记录", "error"));
    });
    document.getElementById("runs-refresh")?.addEventListener("click", () => {
      loadRuns().catch((error) => toast(error.message || "无法刷新运行记录", "error"));
    });
    const page = document.body.dataset.page;
    try {
      if (page === "model-bindings") await loadBindingsPage();
      if (page === "media-assets") await loadAssets();
      if (page === "usage-runs") await loadRuns();
    } catch (error) {
      toast(error.message || "无法加载管理数据", "error");
    }
  });
})();
