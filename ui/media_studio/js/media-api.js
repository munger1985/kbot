/* 多媒体创作工作台公共 API Client：附加 Authorization，不保存下游服务凭据。 */
(function () {
  "use strict";
  const basePath = "/api/v1/apps/media-studio";

  function requestId() {
    return globalThis.KBotMediaAuth?.uuid?.()
      || globalThis.crypto?.randomUUID?.()
      || `media-ui-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function baseUrl() {
    return String(globalThis.KBOT_UI_CONFIG?.mainApiBaseUrl || "").trim().replace(/\/+$/, "");
  }

  function resolveUrl(path) {
    const normalized = String(path || "");
    if (normalized.startsWith("/api/")) return `${baseUrl()}${normalized}`;
    const suffix = normalized.startsWith("/") ? normalized : `/${normalized}`;
    return `${baseUrl()}${basePath}${suffix}`;
  }

  function errorMessage(payload, status) {
    const detail = payload?.detail;
    if (Array.isArray(detail)) return detail.map((item) => item?.msg || "请求内容无效").join("；");
    if (detail && typeof detail === "object") return detail.message || detail.detail || detail.code || `请求失败（HTTP ${status}）`;
    return detail || payload?.message || payload?.code || `请求失败（HTTP ${status}）`;
  }

  function authHeaders() {
    const session = globalThis.KBotMediaAuth?.load?.();
    return session?.access_token ? { Authorization: `Bearer ${session.access_token}` } : {};
  }

  function redirectIfUnauthorized() {
    globalThis.KBotMediaAuth?.clear?.();
    if (document.body?.dataset?.page !== "login") location.replace("./login.html");
  }

  async function decode(response) {
    if (response.status === 204) return null;
    const raw = await response.text();
    if (!raw) return null;
    try { return JSON.parse(raw); } catch (_) { return raw; }
  }

  async function request(path, options = {}) {
    const headers = {
      Accept: "application/json",
      "X-Request-ID": requestId(),
      ...authHeaders(),
      ...(options.headers || {}),
    };
    if (options.body && !(options.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const response = await fetch(resolveUrl(path), {
      credentials: "same-origin",
      cache: "no-store",
      ...options,
      headers,
    });
    const payload = await decode(response);
    if (!response.ok) {
      if (response.status === 401) redirectIfUnauthorized();
      const error = new Error(errorMessage(payload, response.status));
      error.status = response.status;
      error.code = payload?.code || payload?.detail?.code || "MEDIA_STUDIO_REQUEST_FAILED";
      error.requestId = payload?.request_id || response.headers.get("X-Request-ID") || "";
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  async function json(path, method, payload, options = {}) {
    return request(path, {
      ...options,
      method,
      body: payload === undefined ? undefined : JSON.stringify(payload),
    });
  }

  async function requestBlob(path, options = {}) {
    const headers = {
      Accept: "image/*",
      "X-Request-ID": requestId(),
      ...authHeaders(),
      ...(options.headers || {}),
    };
    const response = await fetch(resolveUrl(path), {
      credentials: "same-origin",
      cache: "no-store",
      ...options,
      headers,
    });
    if (!response.ok) {
      const payload = await decode(response);
      if (response.status === 401) redirectIfUnauthorized();
      throw new Error(errorMessage(payload, response.status));
    }
    return response.blob();
  }

  async function stream(path, handlers = {}, signal) {
    const response = await fetch(resolveUrl(path), {
      credentials: "same-origin",
      cache: "no-store",
      headers: {
        Accept: "text/event-stream",
        "Last-Event-ID": String(handlers.lastEventId || 0),
        "X-Request-ID": requestId(),
        ...authHeaders(),
      },
      signal,
    });
    if (!response.ok || !response.body) {
      const payload = await decode(response);
      if (response.status === 401) redirectIfUnauthorized();
      const error = new Error(errorMessage(payload, response.status));
      error.status = response.status;
      error.code = payload?.code || payload?.detail?.code || "MEDIA_STUDIO_EVENT_STREAM_FAILED";
      throw error;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const result = await reader.read();
      buffer += decoder.decode(result.value || new Uint8Array(), { stream: !result.done });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        if (!block || block.startsWith(":")) continue;
        const event = { id: "", type: "message", data: "" };
        block.split(/\r?\n/).forEach((line) => {
          const separator = line.indexOf(":");
          const field = separator < 0 ? line : line.slice(0, separator);
          const value = separator < 0 ? "" : line.slice(separator + 1).replace(/^ /, "");
          if (field === "id") event.id = value;
          if (field === "event") event.type = value;
          if (field === "data") event.data += `${event.data ? "\n" : ""}${value}`;
        });
        try { event.json = event.data ? JSON.parse(event.data) : null; }
        catch (_) { event.json = event.data; }
        handlers.onEvent?.(event);
        if (event.type === "done") return;
      }
      if (result.done) return;
    }
  }

  function withQuery(path, params) {
    const search = new URLSearchParams();
    Object.entries(params || {}).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
    });
    const query = search.toString();
    return query ? `${path}?${query}` : path;
  }

  const ModelCategory = Object.freeze({
    LLM: 1,
    TXT_EMBEDDING: 2,
    IMG_EMBEDDING: 3,
    VLM: 5,
    OCR: 6,
  });

  function items(payload) {
    if (Array.isArray(payload)) return payload;
    if (Array.isArray(payload?.items)) return payload.items;
    if (Array.isArray(payload?.models)) return payload.models;
    return [];
  }

  function modelCategory(row) {
    const raw = String(row?.category ?? row?.model_type ?? row?.model_category ?? "").trim().toUpperCase();
    if (!raw) return NaN;
    if (raw === "EMBEDDING" || raw === "TEXT_EMBEDDING") return ModelCategory.TXT_EMBEDDING;
    if (raw === "IMAGE_EMBEDDING" || raw === "VISUAL_EMBEDDING") return ModelCategory.IMG_EMBEDDING;
    if (ModelCategory[raw] !== undefined) return ModelCategory[raw];
    const numeric = Number(raw);
    return Number.isFinite(numeric) ? numeric : NaN;
  }

  function isActiveModel(row) {
    const status = String(row?.status ?? "").trim().toUpperCase();
    return status === "ACTIVE" || status === "ENABLED" || status === "1";
  }

  globalThis.KBotMediaApi = {
    basePath, json, request, requestBlob, requestId, stream, withQuery,
    ModelCategory, items, modelCategory, isActiveModel,
  };
})();
