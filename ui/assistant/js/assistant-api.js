/* 智能工作台公共 API Client：附加 Authorization，不保存下游服务凭据。 */
(function () {
  "use strict";
  const basePath = "/api/v1/apps/assistant";

  function requestId() {
    return globalThis.KBotAssistantAuth?.uuid?.()
      || globalThis.crypto?.randomUUID?.()
      || `assistant-ui-${Date.now()}-${Math.random().toString(16).slice(2)}`;
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
    const session = globalThis.KBotAssistantAuth?.load?.();
    return session?.access_token ? { Authorization: `Bearer ${session.access_token}` } : {};
  }

  function redirectIfUnauthorized() {
    globalThis.KBotAssistantAuth?.clear?.();
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
      error.code = payload?.code || payload?.detail?.code || "ASSISTANT_REQUEST_FAILED";
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

  function withQuery(path, params) {
    const search = new URLSearchParams();
    Object.entries(params || {}).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
    });
    const query = search.toString();
    return query ? `${path}?${query}` : path;
  }

  globalThis.KBotAssistantApi = {
    basePath, json, request, requestBlob, requestId, withQuery,
  };
})();
