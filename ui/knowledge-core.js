(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  let pollTimer = null;
  let trackingTimer = null;
  let trackingRefreshing = false;
  const TRACKING_STORAGE_KEY = "kbot.kc.upload-tracking.v1";
  const trackedUploads = new Map();

  function loadTrackedUploads() {
    try {
      const rows = JSON.parse(
        localStorage.getItem(TRACKING_STORAGE_KEY) || "[]"
      );
      rows.forEach((item) => trackedUploads.set(item.trackingId, item));
    } catch (_) {
      localStorage.removeItem(TRACKING_STORAGE_KEY);
    }
  }

  function saveTrackedUploads() {
    localStorage.setItem(
      TRACKING_STORAGE_KEY,
      JSON.stringify(Array.from(trackedUploads.values()))
    );
  }

  function isTrackingTerminal(item) {
    if (item.localStage === "FAILED") return true;
    if (item.approvalStatus === "REJECTED") return true;
    if (!["READY", "PARTIAL"].includes(item.revisionStatus)) {
      return item.revisionStatus === "FAILED";
    }
    return item.currentRevisionId === item.revisionId;
  }

  function trackingStage(item) {
    if (item.localStage === "HASHING") return ["计算文件 Hash", ""];
    if (item.localStage === "UPLOADING") return ["上传并创建入库记录", ""];
    if (item.localStage === "FAILED") return ["上传失败", "failed"];
    if (item.approvalStatus === "PENDING") return ["等待人工审批", ""];
    if (
      item.approvalStatus === "REJECTED" ||
      item.revisionStatus === "REJECTED"
    ) {
      return ["审批已驳回", "failed"];
    }
    if (item.revisionStatus === "FAILED") return ["处理失败", "failed"];
    const statuses = (item.members || []).map(
      (member) => member.member_status
    );
    if (
      ["READY", "PARTIAL"].includes(item.revisionStatus) &&
      item.currentRevisionId !== item.revisionId
    ) {
      return ["正在生成检索画像", ""];
    }
    if (
      ["READY", "PARTIAL"].includes(item.revisionStatus) &&
      item.currentRevisionId === item.revisionId
    ) {
      return [
        item.revisionStatus === "READY" ? "已完成，可检索" : "部分完成，可检索",
        "ready",
      ];
    }
    if (statuses.includes("FAILED")) return ["部分文件处理失败", "failed"];
    if (statuses.includes("PARSING")) return ["正在解析", ""];
    if (statuses.includes("INDEXING")) return ["正在生成向量索引", ""];
    if (statuses.length && statuses.every((value) => value === "RECEIVED")) {
      return ["等待 Parser 领取", ""];
    }
    return ["KC 已受理", ""];
  }

  function memberDetails(item) {
    if (!item.members?.length) {
      return item.error
        ? `<span class="tracking-stage failed">${KBotUI.escapeHtml(
            item.error
          )}</span>`
        : "尚未读取到文件状态";
    }
    return item.members
      .map((member) => {
        const name =
          member.declared_name || member.external_document_id || "未命名文件";
        const failure = member.failure_code
          ? ` · ${member.failure_stage || "处理"}: ${member.failure_code}`
          : "";
        return `<span class="tracking-member">${KBotUI.escapeHtml(
          name
        )} · <strong>${KBotUI.escapeHtml(
          member.member_status
        )}</strong>${KBotUI.escapeHtml(failure)}</span>`;
      })
      .join("");
  }

  function renderTrackedUploads() {
    const rows = Array.from(trackedUploads.values()).sort(
      (left, right) => right.createdAt.localeCompare(left.createdAt)
    );
    $("#tracking-rows").innerHTML = rows
      .map((item) => {
        const [stage, stageClass] = trackingStage(item);
        const approval = item.approvalStatus || "等待受理";
        const identifiers = item.bundleId
          ? `<span class="tracking-id" title="${KBotUI.escapeHtml(
              item.bundleId
            )}">B: ${KBotUI.escapeHtml(item.bundleId)}</span>
               <span class="tracking-id" title="${KBotUI.escapeHtml(
                 item.revisionId
               )}">R: ${KBotUI.escapeHtml(item.revisionId)}</span>`
          : "尚未分配";
        return `
          <tr>
            <td>${KBotUI.escapeHtml(item.label)}</td>
            <td><span class="tracking-stage ${stageClass}">${KBotUI.escapeHtml(
              stage
            )}</span></td>
            <td><span class="badge">${KBotUI.escapeHtml(approval)}</span></td>
            <td>${memberDetails(item)}</td>
            <td>${identifiers}</td>
            <td>${KBotUI.escapeHtml(
              item.updatedAt
                ? new Date(item.updatedAt).toLocaleTimeString()
                : "等待刷新"
            )}</td>
          </tr>`;
      })
      .join("");
    KBotUI.setStatus(
      $("#tracking-status"),
      rows.length
        ? `正在跟踪 ${rows.length} 条记录，其中 ${
            rows.filter((item) => !isTrackingTerminal(item)).length
          } 条尚未完成`
        : "暂无上传跟踪记录",
      rows.length ? "ok" : ""
    );
  }

  function updateTracked(trackingId, values) {
    const current = trackedUploads.get(trackingId);
    if (!current) return;
    trackedUploads.set(trackingId, {
      ...current,
      ...values,
      updatedAt: new Date().toISOString(),
    });
    saveTrackedUploads();
    renderTrackedUploads();
  }

  async function refreshTrackedUploads() {
    if (trackingRefreshing) return;
    trackingRefreshing = true;
    const config = KBotUI.loadConfig();
    const candidates = Array.from(trackedUploads.values()).filter(
      (item) =>
        item.bundleId &&
        item.revisionId &&
        !isTrackingTerminal(item) &&
        item.domainId === config.domainId &&
        item.baseUrl === config.baseUrl
    );
    try {
      await Promise.all(
        candidates.map(async (item) => {
          try {
            const [revision, bundle] = await Promise.all([
              KBotUI.api(
                `/api/v1/knowledge/bundles/${item.bundleId}/revisions/${item.revisionId}/members`
              ),
              KBotUI.api(`/api/v1/knowledge/bundles/${item.bundleId}`),
            ]);
            updateTracked(item.trackingId, {
              localStage: "ACCEPTED",
              approvalStatus: revision.approval_status,
              revisionStatus: revision.status,
              reviewedBy: revision.reviewed_by,
              members: revision.members || [],
              bundleStatus: bundle.availability_status,
              currentRevisionId: bundle.current_revision_id,
              error: null,
            });
          } catch (error) {
            updateTracked(item.trackingId, {
              error: error.message,
            });
          }
        })
      );
    } finally {
      trackingRefreshing = false;
    }
  }

  function ensureTrackingPolling() {
    if (trackingTimer) return;
    trackingTimer = setInterval(refreshTrackedUploads, 2000);
  }

  loadTrackedUploads();

  KBotUI.bindAuthForm($("#auth-form"), () => {
    KBotUI.setStatus($("#collection-status"), "连接信息已保存", "ok");
    refreshModels();
  });

  function showResponse(payload) {
    $("#response-output").textContent = KBotUI.json(payload);
  }

  function generateCollectionKey() {
    const timestamp = Date.now().toString(36);
    const random = crypto.randomUUID().replaceAll("-", "").slice(0, 8);
    const value = `collection-${timestamp}-${random}`;
    $("#collection-form").elements.collectionKey.value = value;
    return value;
  }

  $("#generate-collection-key").addEventListener(
    "click",
    generateCollectionKey
  );
  generateCollectionKey();

  const MODEL_CATEGORY = {
    LLM: 1,
    TXT_EMBEDDING: 2,
    IMG_EMBEDDING: 3,
    VLM: 5,
  };

  function modelOption(model) {
    const displayName = model.display_name || model.served_model_name;
    const label = `${displayName}（${model.served_model_name}）`;
    return `<option value="${KBotUI.escapeHtml(
      model.model_id
    )}">${KBotUI.escapeHtml(label)}</option>`;
  }

  function fillModelSelect(select, models, emptyLabel, required) {
    const currentValue = select.value;
    const firstLabel = models.length ? emptyLabel : "暂无可用模型";
    select.innerHTML = [
      `<option value="">${KBotUI.escapeHtml(firstLabel)}</option>`,
      ...models.map(modelOption),
    ].join("");
    if (models.some((model) => model.model_id === currentValue)) {
      select.value = currentValue;
    } else if (required && models.length === 1) {
      select.value = models[0].model_id;
    }
    select.disabled = models.length === 0;
  }

  async function refreshModels() {
    const form = $("#collection-form");
    const status = $("#model-status");
    KBotUI.setStatus(status, "正在读取模型目录…");
    try {
      const models = await KBotUI.api("/api/v1/model-catalog", {
        domainOptional: true,
      });
      const byCategory = (category) =>
        models.filter((model) => Number(model.category) === category);
      const llmModels = byCategory(MODEL_CATEGORY.LLM);
      fillModelSelect(
        form.elements.parserLlm,
        llmModels,
        "请选择解析 LLM",
        true
      );
      fillModelSelect(
        form.elements.retrievalLlm,
        llmModels,
        "请选择检索 LLM",
        true
      );
      fillModelSelect(
        form.elements.embedding,
        byCategory(MODEL_CATEGORY.TXT_EMBEDDING),
        "请选择文本 Embedding",
        true
      );
      fillModelSelect(
        form.elements.parserVlm,
        byCategory(MODEL_CATEGORY.VLM),
        "不使用 VLM",
        false
      );
      fillModelSelect(
        form.elements.visualEmbedding,
        byCategory(MODEL_CATEGORY.IMG_EMBEDDING),
        "不使用视觉 Embedding",
        false
      );
      KBotUI.setStatus(
        status,
        `已加载 ${models.length} 个启用模型`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus(status, error.message, "error");
    }
  }

  $("#refresh-models").addEventListener("click", refreshModels);
  refreshModels();

  $("#domain-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const status = $("#domain-status");
    KBotUI.setStatus(status, "正在创建 Domain…");
    try {
      const payload = await KBotUI.api("/api/v1/domains", {
        method: "POST",
        domainOptional: true,
        body: JSON.stringify({
          name: form.elements.name.value.trim(),
          description: form.elements.description.value.trim() || null,
        }),
      });
      const authForm = $("#auth-form");
      authForm.elements.domainId.value = String(payload.domain_id);
      KBotUI.readAuthForm(authForm);
      showResponse(payload);
      KBotUI.setStatus(
        status,
        `Domain 创建成功，当前 Domain ID：${payload.domain_id}`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus(status, error.message, "error");
      showResponse(error.payload || { error: error.message });
    }
  });

  async function refreshCollections() {
    const status = $("#collection-status");
    KBotUI.setStatus(status, "正在读取…");
    try {
      const payload = await KBotUI.api("/api/v1/knowledge/collections");
      const rows = Array.isArray(payload)
        ? payload
        : payload?.collections || payload?.items || [];
      $("#collection-rows").innerHTML = rows
        .map(
          (item) => `
            <tr data-key="${KBotUI.escapeHtml(item.collection_key)}">
              <td>${KBotUI.escapeHtml(item.collection_key)}</td>
              <td>${KBotUI.escapeHtml(item.display_name)}</td>
              <td><span class="badge">${KBotUI.escapeHtml(item.status)}</span></td>
              <td>${KBotUI.escapeHtml(item.collection_id)}</td>
            </tr>`
        )
        .join("");
      $("#collection-rows").querySelectorAll("tr").forEach((row) => {
        row.addEventListener("click", () => {
          $("#collection-rows .selected")?.classList.remove("selected");
          row.classList.add("selected");
          const key = row.dataset.key;
          $("#upload-form").elements.collectionKey.value = key;
          $("#approval-form").elements.collectionKey.value = key;
          $("#binding-form").elements.collectionKey.value = key;
        });
      });
      showResponse(payload);
      KBotUI.setStatus(status, `已读取 ${rows.length} 个 Collection`, "ok");
    } catch (error) {
      KBotUI.setStatus(status, error.message, "error");
    }
  }

  $("#refresh-collections").addEventListener("click", refreshCollections);

  $("#collection-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const status = $("#create-status");
    const optional = (name) => form.elements[name].value.trim() || null;
    KBotUI.setStatus(status, "正在创建…");
    try {
      const payload = await KBotUI.api("/api/v1/knowledge/collections", {
        method: "POST",
        body: JSON.stringify({
          collection_key: form.elements.collectionKey.value.trim(),
          display_name: form.elements.displayName.value.trim(),
          parser_llm_model_id: form.elements.parserLlm.value.trim(),
          parser_vlm_model_id: optional("parserVlm"),
          retrieval_llm_model_id: form.elements.retrievalLlm.value.trim(),
          embedding_model_id: form.elements.embedding.value.trim(),
          visual_embedding_model_id: optional("visualEmbedding"),
          description: optional("description"),
          default_security_level: 1,
          metadata: {},
        }),
      });
      showResponse(payload);
      KBotUI.setStatus(status, "Collection 创建成功", "ok");
      await refreshCollections();
    } catch (error) {
      KBotUI.setStatus(status, error.message, "error");
      showResponse(error.payload || { error: error.message });
    }
  });

  $("#upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const files = Array.from(form.elements.files.files || []);
    const status = $("#upload-status");
    if (!files.length) return;
    const mode = form.elements.groupingMode.value;
    const connection = KBotUI.loadConfig();
    const batchId = crypto.randomUUID();
    const fileSeeds = files.map((file, index) => ({
      file,
      index,
      clientFileId: crypto.randomUUID(),
    }));
    const trackingIds =
      mode === "SINGLE_BUNDLE"
        ? [`${batchId}:bundle`]
        : fileSeeds.map((seed) => `${batchId}:${seed.clientFileId}`);
    trackingIds.forEach((trackingId, index) => {
      const labels =
        mode === "SINGLE_BUNDLE"
          ? files.map((file) => file.name).join("、")
          : files[index].name;
      trackedUploads.set(trackingId, {
        trackingId,
        label: labels,
        collectionKey: form.elements.collectionKey.value.trim(),
        domainId: connection.domainId,
        baseUrl: connection.baseUrl,
        localStage: "HASHING",
        approvalStatus: null,
        revisionStatus: null,
        members: [],
        bundleId: null,
        revisionId: null,
        currentRevisionId: null,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      });
    });
    saveTrackedUploads();
    renderTrackedUploads();
    ensureTrackingPolling();
    KBotUI.setStatus(status, `正在计算 ${files.length} 个文件的 SHA-256…`);
    try {
      const declarations = await Promise.all(
        fileSeeds.map(async ({ file, index, clientFileId }) => ({
          part_name: `file_${index + 1}`,
          client_file_id: clientFileId,
          display_name: file.name,
          declared_mime_type: file.type || "application/octet-stream",
          byte_size: file.size,
          content_sha256: await KBotUI.sha256(file),
          ordinal: index,
          role: "CONTENT",
          required_flag: false,
        }))
      );
      trackingIds.forEach((trackingId) =>
        updateTracked(trackingId, { localStage: "UPLOADING" })
      );
      const data = new FormData();
      data.append("grouping_mode", mode);
      data.append("files", JSON.stringify(declarations));
      declarations.forEach((item, index) => {
        data.append(item.part_name, files[index], files[index].name);
      });
      if (mode === "SINGLE_BUNDLE") {
        const title = form.elements.bundleTitle.value.trim();
        if (!title) throw new Error("SINGLE_BUNDLE 必须填写 Bundle 标题");
        data.append(
          "bundle",
          JSON.stringify({
            client_bundle_id: crypto.randomUUID(),
            title,
            security_level: Number(form.elements.securityLevel.value || 1),
            facet: {},
            metadata: { source: "kc-test-ui" },
          })
        );
      }
      KBotUI.setStatus(status, "正在上传并等待 KC 受理…");
      const payload = await KBotUI.api(
        `/api/v1/knowledge/collections/${encodeURIComponent(
          form.elements.collectionKey.value.trim()
        )}/ingestions/user-files`,
        {
          method: "POST",
          headers: { "Idempotency-Key": KBotUI.idempotency("ui-upload") },
          body: data,
        }
      );
      showResponse(payload);
      if (mode === "SINGLE_BUNDLE") {
        const item = (payload.items || [])[0] || {};
        updateTracked(trackingIds[0], {
          localStage: item.bundle_id ? "ACCEPTED" : "FAILED",
          bundleId: item.bundle_id || null,
          revisionId: item.bundle_revision_id || null,
          approvalStatus:
            item.status === "PENDING_REVIEW" ? "PENDING" : null,
          error: item.error || item.message || null,
        });
      } else {
        declarations.forEach((declaration, index) => {
          const item = (payload.items || []).find(
            (candidate) =>
              candidate.client_file_id === declaration.client_file_id
          ) || {};
          updateTracked(trackingIds[index], {
            localStage: item.bundle_id ? "ACCEPTED" : "FAILED",
            bundleId: item.bundle_id || null,
            revisionId: item.bundle_revision_id || null,
            approvalStatus:
              item.status === "PENDING_REVIEW" ? "PENDING" : null,
            error: item.error || item.message || null,
          });
        });
      }
      const received = (payload.items || []).find(
        (item) => item.bundle_id
      );
      if (received) {
        $("#status-form").elements.bundleId.value = received.bundle_id || "";
        $("#status-form").elements.revisionId.value =
          received.bundle_revision_id || "";
      }
      $("#approval-form").elements.collectionKey.value =
        form.elements.collectionKey.value.trim();
      const pendingCount = (payload.items || []).filter(
        (item) => item.status === "PENDING_REVIEW"
      ).length;
      KBotUI.setStatus(
        status,
        pendingCount
          ? `上传完成，${pendingCount} 个 Bundle 等待人工审批`
          : "KC 已受理上传请求",
        "ok"
      );
      if (pendingCount) await refreshApprovals();
      await refreshTrackedUploads();
    } catch (error) {
      trackingIds.forEach((trackingId) =>
        updateTracked(trackingId, {
          localStage: "FAILED",
          error: error.message,
        })
      );
      KBotUI.setStatus(status, error.message, "error");
      showResponse(error.payload || { error: error.message });
    }
  });

  async function reviewApproval(bundleRevisionId, decision) {
    const form = $("#approval-form");
    const collectionKey = form.elements.collectionKey.value.trim();
    const comment = form.elements.comment.value.trim() || null;
    const status = $("#approval-status");
    KBotUI.setStatus(
      status,
      decision === "APPROVE" ? "正在批准并创建解析任务…" : "正在驳回…"
    );
    try {
      const payload = await KBotUI.api(
        `/api/v1/knowledge/collections/${encodeURIComponent(
          collectionKey
        )}/bundle-revisions/${bundleRevisionId}/approval`,
        {
          method: "POST",
          body: JSON.stringify({ decision, comment }),
        }
      );
      showResponse(payload);
      KBotUI.setStatus(
        status,
        decision === "APPROVE"
          ? "审批通过，解析任务已创建"
          : "已驳回，本 Revision 不会解析",
        "ok"
      );
      Array.from(trackedUploads.values())
        .filter((item) => item.revisionId === bundleRevisionId)
        .forEach((tracked) => updateTracked(tracked.trackingId, {
          approvalStatus:
            decision === "APPROVE" ? "APPROVED" : "REJECTED",
          revisionStatus:
            decision === "APPROVE" ? "PROCESSING" : "REJECTED",
        }));
      await refreshTrackedUploads();
      await refreshApprovals();
    } catch (error) {
      KBotUI.setStatus(status, error.message, "error");
      showResponse(error.payload || { error: error.message });
    }
  }

  async function refreshApprovals() {
    const form = $("#approval-form");
    const collectionKey = form.elements.collectionKey.value.trim();
    const status = $("#approval-status");
    if (!collectionKey) {
      KBotUI.setStatus(status, "请先填写 collection_key", "error");
      return;
    }
    KBotUI.setStatus(status, "正在读取待审批列表…");
    try {
      const payload = await KBotUI.api(
        `/api/v1/knowledge/collections/${encodeURIComponent(
          collectionKey
        )}/approvals`
      );
      const rows = payload.items || [];
      $("#approval-rows").innerHTML = rows
        .map((item) => {
          const names = (item.document_names || []).join("、");
          const title = names
            ? `${item.title}：${names}`
            : item.title;
          return `
            <tr data-revision-id="${KBotUI.escapeHtml(
              item.bundle_revision_id
            )}">
              <td>${KBotUI.escapeHtml(title)}</td>
              <td>${KBotUI.escapeHtml(item.bundle_revision_id)}</td>
              <td>
                <button type="button" data-decision="APPROVE">批准</button>
                <button type="button" data-decision="REJECT">驳回</button>
              </td>
            </tr>`;
        })
        .join("");
      $("#approval-rows")
        .querySelectorAll("button[data-decision]")
        .forEach((button) => {
          button.addEventListener("click", () => {
            const row = button.closest("tr");
            reviewApproval(row.dataset.revisionId, button.dataset.decision);
          });
        });
      showResponse(payload);
      KBotUI.setStatus(
        status,
        `当前有 ${rows.length} 个 Bundle 等待审批`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus(status, error.message, "error");
      showResponse(error.payload || { error: error.message });
    }
  }

  $("#approval-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await refreshApprovals();
  });

  $("#refresh-tracking").addEventListener("click", async () => {
    await refreshTrackedUploads();
  });

  $("#clear-finished-tracking").addEventListener("click", () => {
    Array.from(trackedUploads.entries()).forEach(([trackingId, item]) => {
      if (isTrackingTerminal(item)) trackedUploads.delete(trackingId);
    });
    saveTrackedUploads();
    renderTrackedUploads();
  });

  renderTrackedUploads();
  ensureTrackingPolling();
  refreshTrackedUploads();

  async function readBundleStatus() {
    const form = $("#status-form");
    const bundleId = form.elements.bundleId.value.trim();
    const revisionId = form.elements.revisionId.value.trim();
    const path = revisionId
      ? `/api/v1/knowledge/bundles/${bundleId}/revisions/${revisionId}/members`
      : `/api/v1/knowledge/bundles/${bundleId}`;
    const payload = await KBotUI.api(path);
    $("#bundle-output").textContent = KBotUI.json(payload);
    showResponse(payload);
    KBotUI.setStatus(
      $("#bundle-status"),
      `最近刷新：${new Date().toLocaleTimeString()}`,
      "ok"
    );
    return payload;
  }

  $("#status-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      await readBundleStatus();
    } catch (error) {
      KBotUI.setStatus($("#bundle-status"), error.message, "error");
    }
  });

  $("#poll-status").addEventListener("click", async () => {
    clearInterval(pollTimer);
    try {
      await readBundleStatus();
    } catch (error) {
      KBotUI.setStatus($("#bundle-status"), error.message, "error");
    }
    pollTimer = setInterval(async () => {
      try {
        await readBundleStatus();
      } catch (error) {
        KBotUI.setStatus($("#bundle-status"), error.message, "error");
      }
    }, 2000);
  });

  $("#stop-poll").addEventListener("click", () => {
    clearInterval(pollTimer);
    pollTimer = null;
    KBotUI.setStatus($("#bundle-status"), "轮询已停止");
  });

  $("#binding-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const agentId = form.elements.agentId.value.trim();
    const collectionKey = form.elements.collectionKey.value.trim();
    try {
      const payload = await KBotUI.api(
        `/api/v1/knowledge/agents/${agentId}/collections/${encodeURIComponent(
          collectionKey
        )}/binding`,
        { method: "PUT", body: JSON.stringify({ note: "kc-test-ui" }) }
      );
      showResponse(payload);
      KBotUI.setStatus($("#binding-status"), "绑定成功", "ok");
    } catch (error) {
      KBotUI.setStatus($("#binding-status"), error.message, "error");
    }
  });

  $("#list-bindings").addEventListener("click", async () => {
    const agentId = $("#binding-form").elements.agentId.value.trim();
    try {
      const payload = await KBotUI.api(
        `/api/v1/knowledge/agents/${agentId}/collection-bindings`
      );
      showResponse(payload);
      KBotUI.setStatus($("#binding-status"), "绑定列表已读取", "ok");
    } catch (error) {
      KBotUI.setStatus($("#binding-status"), error.message, "error");
    }
  });
})();
