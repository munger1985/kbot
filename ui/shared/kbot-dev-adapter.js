/* KM 独立页面认证适配器：登录换取短期 Token，后续请求统一携带 Bearer Token。 */
(function () {
  "use strict";

  const connectionKey = "kbot.km.connection.v2";
  const sessionKey = "kbot.km.session.v1";

  function loadConnection() {
    let value = {};
    try { value = JSON.parse(localStorage.getItem(connectionKey) || "{}"); } catch (_) { value = {}; }
    return {
      baseUrl: String(value.baseUrl || `${location.protocol}//${location.hostname}:18099`).replace(/\/+$/, ""),
    };
  }

  function saveConnection(value) {
    localStorage.setItem(connectionKey, JSON.stringify({
      baseUrl: String(value.baseUrl || "").replace(/\/+$/, ""),
    }));
    return loadConnection();
  }

  function loadSession() {
    try { return JSON.parse(sessionStorage.getItem(sessionKey) || "null"); } catch (_) { return null; }
  }

  function saveSession(value) {
    sessionStorage.setItem(sessionKey, JSON.stringify(value));
    return value;
  }

  function clearSession() {
    sessionStorage.removeItem(sessionKey);
  }

  function tokenExpired(session) {
    return !session?.access_token || !session?.expires_at
      || new Date(session.expires_at).getTime() <= Date.now();
  }

  function requireSession() {
    const session = loadSession();
    if (tokenExpired(session)) {
      clearSession();
      if (!location.pathname.endsWith("/login.html")) location.replace("./login.html");
      throw new Error("KM 登录已过期，请重新登录");
    }
    return session;
  }

  async function decode(response) {
    const text = await response.text();
    if (!text) return null;
    try { return JSON.parse(text); } catch (_) { return text; }
  }

  function failure(response, payload) {
    const detail = payload?.detail;
    const message = typeof detail === "string"
      ? detail : detail?.message || payload?.message || payload?.title
        || `请求失败（HTTP ${response.status}）`;
    const error = new Error(message);
    error.status = response.status;
    error.code = payload?.code || payload?.detail?.code || "KM_REQUEST_FAILED";
    error.requestId = payload?.request_id || response.headers.get("X-Request-ID") || "";
    return error;
  }

  async function raw(path, options = {}, token = "") {
    const headers = {
      Accept: "application/json",
      "X-Request-ID": KBotKmApi.uuid(),
      ...(options.headers || {}),
    };
    if (token) headers.Authorization = `Bearer ${token}`;
    if (options.body && !(options.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const response = await fetch(`${loadConnection().baseUrl}${path}`, { ...options, headers });
    const payload = await decode(response);
    if (!response.ok) throw failure(response, payload);
    return payload;
  }

  async function login(payload) {
    const result = await raw("/api/v1/apps/km-asset/auth/login", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    return saveSession(result);
  }

  async function changePassword(payload) {
    const session = requireSession();
    const result = await raw("/api/v1/apps/km-asset/auth/password", {
      method: "POST",
      body: JSON.stringify(payload),
    }, session.access_token);
    return saveSession({ ...session, ...result });
  }

  async function request(path, options = {}) {
    const session = requireSession();
    try {
      return await raw(path, options, session.access_token);
    } catch (error) {
      if (error.status === 401) {
        clearSession();
        if (!location.pathname.endsWith("/login.html")) location.replace("./login.html");
      }
      throw error;
    }
  }

  async function blob(path, options = {}) {
    const session = requireSession();
    const headers = {
      Authorization: `Bearer ${session.access_token}`,
      "X-Request-ID": KBotKmApi.uuid(),
      ...(options.headers || {}),
    };
    const response = await fetch(`${loadConnection().baseUrl}${path}`, { ...options, headers });
    if (!response.ok) throw failure(response, await decode(response));
    return { data: await response.blob(), contentType: response.headers.get("Content-Type") || "application/octet-stream" };
  }

  async function stream(path, handlers = {}, signal) {
    const session = requireSession();
    const response = await fetch(`${loadConnection().baseUrl}${path}`, {
      headers: {
        Authorization: `Bearer ${session.access_token}`,
        Accept: "text/event-stream",
        "Last-Event-ID": String(handlers.lastEventId || 0),
        "X-Request-ID": KBotKmApi.uuid(),
      },
      signal,
    });
    if (!response.ok || !response.body) throw failure(response, await decode(response));
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = "";
    while (true) {
      const result = await reader.read(); buffer += decoder.decode(result.value || new Uint8Array(), { stream: !result.done });
      const blocks = buffer.split(/\r?\n\r?\n/); buffer = blocks.pop() || "";
      for (const block of blocks) {
        if (!block || block.startsWith(":")) continue;
        const item = { id: "", type: "message", data: "" };
        block.split(/\r?\n/).forEach((line) => { const at = line.indexOf(":"); const key = at < 0 ? line : line.slice(0, at); const value = at < 0 ? "" : line.slice(at + 1).replace(/^ /, ""); if (key === "id") item.id = value; if (key === "event") item.type = value; if (key === "data") item.data += `${item.data ? "\n" : ""}${value}`; });
        try { item.json = item.data ? JSON.parse(item.data) : null; } catch (_) { item.json = item.data; }
        handlers.onEvent?.(item); if (item.type === "done") return;
      }
      if (result.done) return;
    }
  }

  KBotKmApi.configure({ request, blob, stream });
  window.KBotKmAuth = {
    changePassword, clearSession, loadConnection, loadSession, login,
    requireSession, saveConnection,
  };
})();
