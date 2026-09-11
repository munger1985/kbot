/* 智能工作台公共 API Client：统一附加请求标识，不保存下游服务凭据。 */
(function () {
  "use strict";
  const basePath = "/api/v1/apps/assistant";

  function requestId() {
    return globalThis.crypto?.randomUUID?.() || `assistant-ui-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function baseUrl() {
    return String(globalThis.KBOT_UI_CONFIG?.mainApiBaseUrl || "").trim().replace(/\/+$/, "");
  }

  function errorMessage(payload, status) {
    const detail = payload?.detail;
    if (Array.isArray(detail)) return detail.map((item) => item?.msg || "请求内容无效").join("；");
    if (detail && typeof detail === "object") return detail.message || detail.detail || detail.code || `请求失败（HTTP ${status}）`;
    return detail || payload?.message || payload?.code || `请求失败（HTTP ${status}）`;
  }

  async function decode(response) {
    if (response.status === 204) return null;
    const raw = await response.text();
    if (!raw) return null;
    try { return JSON.parse(raw); } catch (_) { return raw; }
  }

  async function request(path, options = {}) {
    const headers = { Accept: "application/json", "X-Request-ID": requestId(), ...(options.headers || {}) };
    if (options.body && !(options.body instanceof FormData) && !headers["Content-Type"]) headers["Content-Type"] = "application/json";
    const response = await fetch(`${baseUrl()}${basePath}${path}`, { credentials: "same-origin", cache: "no-store", ...options, headers });
    const payload = await decode(response);
    if (!response.ok) {
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
    return request(path, { ...options, method, body: payload === undefined ? undefined : JSON.stringify(payload) });
  }

  async function stream(path, handlers = {}, signal) {
    const response = await fetch(`${baseUrl()}${basePath}${path}`, { credentials: "same-origin", cache: "no-store", signal, headers: { Accept: "text/event-stream", "Last-Event-ID": String(handlers.lastEventId || 0), "X-Request-ID": requestId() } });
    if (!response.ok || !response.body) throw new Error(`事件流连接失败（HTTP ${response.status}）`);
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
        block.split(/\r?\n/).forEach((line) => { const index = line.indexOf(":"); const field = index < 0 ? line : line.slice(0, index); const value = index < 0 ? "" : line.slice(index + 1).replace(/^ /, ""); if (field === "id") event.id = value; if (field === "event") event.type = value; if (field === "data") event.data += `${event.data ? "\n" : ""}${value}`; });
        try { event.json = event.data ? JSON.parse(event.data) : null; } catch (_) { event.json = event.data; }
        handlers.onEvent?.(event);
        if (event.type === "done") return;
      }
      if (result.done) return;
    }
  }

  globalThis.KBotAssistantApi = { basePath, json, request, requestId, stream };
})();
