/* KBot KM 正式页面公开 API Client。身份与 API Key 必须由 APEX 或同源网关注入。 */
(function () {
  "use strict";

  let adapter = null;

  function configure(nextAdapter) {
    adapter = nextAdapter || null;
  }

  function uuid() {
    return globalThis.crypto?.randomUUID?.()
      || `ui-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function query(values) {
    const params = new URLSearchParams();
    Object.entries(values || {}).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") {
        params.set(key, String(value));
      }
    });
    const encoded = params.toString();
    return encoded ? `?${encoded}` : "";
  }

  function message(payload, fallback) {
    if (!payload) return fallback;
    const detail = payload.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object") {
      return detail.message || detail.detail || detail.code || fallback;
    }
    return payload.message || payload.title || payload.code || fallback;
  }

  async function decode(response) {
    if (response.status === 204) return null;
    const text = await response.text();
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch (_) {
      return text;
    }
  }

  async function request(path, options = {}) {
    if (adapter?.request) return adapter.request(path, options);
    if (globalThis.KBotApi?.request) {
      return globalThis.KBotApi.request(path, {
        label: options.label || "KM 请求",
        ...options,
      });
    }
    const headers = {
      Accept: "application/json",
      "X-Request-ID": uuid(),
      ...(options.headers || {}),
    };
    const body = options.body;
    if (body && !(body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const response = await fetch(path, {
      credentials: "same-origin",
      ...options,
      headers,
    });
    const payload = await decode(response);
    if (!response.ok) {
      const error = new Error(message(payload, `请求失败（HTTP ${response.status}）`));
      error.status = response.status;
      error.code = payload?.code || payload?.detail?.code || "KM_REQUEST_FAILED";
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

  async function blob(path, options = {}) {
    if (adapter?.blob) return adapter.blob(path, options);
    const response = await fetch(path, {
      credentials: "same-origin",
      ...options,
      headers: { "X-Request-ID": uuid(), ...(options.headers || {}) },
    });
    if (!response.ok) {
      const payload = await decode(response);
      throw new Error(message(payload, `文件读取失败（HTTP ${response.status}）`));
    }
    return { data: await response.blob(), contentType: response.headers.get("Content-Type") || "application/octet-stream" };
  }

  async function stream(path, handlers = {}, signal) {
    if (adapter?.stream) return adapter.stream(path, handlers, signal);
    const response = await fetch(path, {
      credentials: "same-origin",
      headers: {
        Accept: "text/event-stream",
        "Last-Event-ID": String(handlers.lastEventId || 0),
        "X-Request-ID": uuid(),
      },
      signal,
    });
    if (!response.ok || !response.body) {
      const payload = await decode(response);
      throw new Error(message(payload, `事件流连接失败（HTTP ${response.status}）`));
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

  function items(payload) {
    if (Array.isArray(payload)) return payload;
    return Array.isArray(payload?.items) ? payload.items : [];
  }

  globalThis.KBotKmApi = { blob, configure, items, json, query, request, stream, uuid };
})();
