/* KBot 4.0 本地测试页共享工具。仅使用开发环境认证绕过。 */
(function () {
  "use strict";

  const STORAGE_KEY = "kbot.ui.connection.v1";

  function defaultBaseUrl() {
    const hostname = window.location.hostname;
    if (hostname && hostname !== "127.0.0.1" && hostname !== "localhost") {
      return `${window.location.protocol}//${hostname}:18099`;
    }
    return "http://127.0.0.1:18099";
  }

  function uuid() {
    // 测试页需要兼容 HTTP 环境；该值只作请求关联与幂等标识。
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(
      /[xy]/g,
      (character) => {
        const value = Math.floor(Math.random() * 16);
        const nibble = character === "x" ? value : (value & 0x3) | 0x8;
        return nibble.toString(16);
      }
    );
  }

  function loadConfig() {
    let persisted = {};
    try {
      persisted = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    } catch (_) {
      persisted = {};
    }
    return {
      baseUrl: persisted.baseUrl || defaultBaseUrl(),
      domainId: persisted.domainId || "",
      userId: persisted.userId || "ui-tester",
    };
  }

  function saveConfig(config) {
    const baseUrl = String(config.baseUrl || "").replace(/\/+$/, "");
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        baseUrl,
        domainId: String(config.domainId || "").trim(),
        userId: String(config.userId || "").trim(),
      })
    );
    return loadConfig();
  }

  function readAuthForm(form) {
    return saveConfig({
      baseUrl: form.elements.baseUrl.value,
      domainId: form.elements.domainId.value,
      userId: form.elements.userId.value,
    });
  }

  function bindAuthForm(form, onSaved) {
    const config = loadConfig();
    for (const key of ["baseUrl", "domainId", "userId"]) {
      if (form.elements[key]) {
        form.elements[key].value = config[key];
      }
    }
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const updated = readAuthForm(form);
      if (typeof onSaved === "function") onSaved(updated);
    });
  }

  function requestHeaders(config, extra) {
    const headers = {
      "X-KBot-Test-Auth": "true",
      "X-KBot-User-ID": config.userId,
      "X-Request-ID": uuid(),
      ...(extra || {}),
    };
    if (config.domainId) {
      headers["X-KBot-Domain-ID"] = config.domainId;
    }
    return headers;
  }

  async function api(path, options) {
    const config = loadConfig();
    const domainOptional = Boolean(options?.domainOptional);
    if ((!domainOptional && !config.domainId) || !config.userId) {
      throw new Error("请先填写并保存 Domain ID 和 User ID");
    }
    const requestOptions = { ...(options || {}) };
    delete requestOptions.domainOptional;
    const body = requestOptions.body;
    const headers = requestHeaders(config, requestOptions.headers);
    if (body && !(body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const response = await fetch(`${config.baseUrl}${path}`, {
      ...requestOptions,
      headers,
    });
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (_) {
        payload = text;
      }
    }
    if (!response.ok) {
      const detail =
        payload?.detail || payload?.message || payload?.code || text;
      const error = new Error(
        `HTTP ${response.status}: ${
          typeof detail === "string" ? detail : JSON.stringify(detail)
        }`
      );
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  async function streamSse(path, handlers, signal) {
    const config = loadConfig();
    const response = await fetch(`${config.baseUrl}${path}`, {
      headers: requestHeaders(config, {
        Accept: "text/event-stream",
        "Last-Event-ID": String(handlers.lastEventId || 0),
      }),
      signal,
    });
    if (!response.ok || !response.body) {
      throw new Error(`SSE 连接失败：HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() || "";
      for (const block of blocks) {
        if (!block || block.startsWith(":")) continue;
        const event = { id: "", type: "message", data: "" };
        for (const line of block.split(/\r?\n/)) {
          const separator = line.indexOf(":");
          const field = separator < 0 ? line : line.slice(0, separator);
          const valueText =
            separator < 0 ? "" : line.slice(separator + 1).replace(/^ /, "");
          if (field === "id") event.id = valueText;
          if (field === "event") event.type = valueText;
          if (field === "data") {
            event.data += `${event.data ? "\n" : ""}${valueText}`;
          }
        }
        try {
          event.json = event.data ? JSON.parse(event.data) : null;
        } catch (_) {
          event.json = event.data;
        }
        if (handlers.onEvent) handlers.onEvent(event);
        if (event.type === "done") return;
      }
      if (done) return;
    }
  }

  function setStatus(element, message, kind) {
    element.textContent = message || "";
    element.className = `status${kind ? ` ${kind}` : ""}`;
  }

  function json(value) {
    return JSON.stringify(value, null, 2);
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function idempotency(prefix) {
    return `${prefix}-${Date.now()}-${uuid()}`;
  }

  async function sha256(file) {
    const bytes = await file.arrayBuffer();
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }

  window.KBotUI = {
    api,
    bindAuthForm,
    escapeHtml,
    idempotency,
    json,
    loadConfig,
    readAuthForm,
    setStatus,
    sha256,
    streamSse,
    uuid,
  };
})();
