/* KM 独立页面认证适配器；独立文件名避免旧代理缓存继续执行过期登出逻辑。 */
(function () {
  "use strict";

  const sessionKey = "kbot.km.session.v1";
  const authFailureKey = "kbot.km.last-auth-failure.v1";

  function mainApiBaseUrl() {
    const value = String(globalThis.KBOT_UI_CONFIG?.mainApiBaseUrl || "")
      .trim().replace(/\/+$/, "");
    if (!value) throw new Error("KM UI 未加载 Main API 部署配置");
    return value;
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

  function recordAuthFailure(value) {
    try {
      localStorage.setItem(authFailureKey, JSON.stringify({
        occurred_at: new Date().toISOString(),
        ...value,
      }));
    } catch (_) { /* 浏览器禁用持久化时不影响主流程。 */ }
  }

  function loadAuthFailure() {
    try { return JSON.parse(localStorage.getItem(authFailureKey) || "null"); }
    catch (_) { return null; }
  }

  function clearAuthFailure() {
    try { localStorage.removeItem(authFailureKey); } catch (_) { /* 无需处理。 */ }
  }

  function requireSession() {
    const session = loadSession();
    // Token 是否过期由 Main API 的签名与 exp 校验决定，避免浏览器时钟偏差
    // 或序列化差异在页面切换时提前销毁仍有效的登录态。
    if (!session?.access_token) {
      recordAuthFailure({
        path: location.pathname,
        code: "KM_SESSION_MISSING",
        message: "浏览器中没有 KM 登录 Session",
      });
      if (!location.pathname.endsWith("/login.html")) location.replace("./login.html");
      throw new Error("请先登录 KM Asset");
    }
    return session;
  }

  async function decode(response) {
    const text = await response.text();
    if (!text) return null;
    try { return JSON.parse(text); } catch (_) { return text; }
  }

  function failure(response, payload, path = "") {
    const detail = payload?.detail;
    const message = typeof detail === "string"
      ? detail : detail?.message || payload?.message || payload?.title
        || `请求失败（HTTP ${response.status}）`;
    const error = new Error(message);
    error.status = response.status;
    error.code = payload?.code || payload?.detail?.code || "KM_REQUEST_FAILED";
    error.requestId = payload?.request_id || response.headers.get("X-Request-ID") || "";
    error.payload = payload;
    error.path = path;
    return error;
  }

  function sessionIsInvalid(error) {
    return error?.status === 401 && [
      "INVALID_KM_TOKEN",
      "KM_TOKEN_EXPIRED",
    ].includes(String(error?.code || ""));
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
    const response = await fetch(`${mainApiBaseUrl()}${path}`, { ...options, headers });
    const payload = await decode(response);
    if (!response.ok) throw failure(response, payload, path);
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
      // 网络认证错误必须留在当前页面展示，不能自动清除 Session 或跳转，
      // 否则会把 Main API 路由、内部凭据和 Token 错误混成“回到登录页”。
      if (error?.status === 401) {
        recordAuthFailure({
          path,
          status: error.status,
          code: error.code,
          message: error.message,
          request_id: error.requestId,
          session_invalid: sessionIsInvalid(error),
        });
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
    const response = await fetch(`${mainApiBaseUrl()}${path}`, { ...options, headers });
    if (!response.ok) throw failure(response, await decode(response), path);
    return { data: await response.blob(), contentType: response.headers.get("Content-Type") || "application/octet-stream" };
  }

  async function stream(path, handlers = {}, signal) {
    const session = requireSession();
    const response = await fetch(`${mainApiBaseUrl()}${path}`, {
      headers: {
        Authorization: `Bearer ${session.access_token}`,
        Accept: "text/event-stream",
        "Last-Event-ID": String(handlers.lastEventId || 0),
        "X-Request-ID": KBotKmApi.uuid(),
      },
      signal,
    });
    if (!response.ok || !response.body) throw failure(response, await decode(response), path);
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
    changePassword, clearAuthFailure, clearSession, loadAuthFailure,
    loadSession, login, recordAuthFailure, requireSession, sessionIsInvalid,
  };
})();
