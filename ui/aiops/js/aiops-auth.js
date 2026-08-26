(function () {
  "use strict";

  const storageKey = "kbot.aiops.session.v1";
  const baseUrl = () => String(
    globalThis.KBOT_UI_CONFIG?.mainApiBaseUrl || "",
  ).replace(/\/+$/, "");

  function uuid() {
    return globalThis.crypto?.randomUUID?.()
      || `ui-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function load() {
    try {
      return JSON.parse(sessionStorage.getItem(storageKey) || "null");
    } catch (_) {
      return null;
    }
  }

  function save(value) {
    sessionStorage.setItem(storageKey, JSON.stringify(value));
    return value;
  }

  const clear = () => sessionStorage.removeItem(storageKey);

  function errorMessage(payload, status) {
    const detail = payload?.detail;
    if (Array.isArray(detail)) {
      const messages = detail.map((item) => {
        const location = Array.isArray(item?.loc)
          ? item.loc.filter((value) => value !== "body").join(".")
          : "";
        return `${location ? `${location}：` : ""}${item?.msg || "请求内容无效"}`;
      });
      return messages.join("；");
    }
    if (detail && typeof detail === "object") {
      return detail.message || detail.detail || detail.code || `请求失败（HTTP ${status}）`;
    }
    return detail || payload?.message || payload?.code || `请求失败（HTTP ${status}）`;
  }

  async function raw(path, options = {}, token = "") {
    const headers = {
      Accept: "application/json",
      "X-Request-ID": uuid(),
      ...(options.headers || {}),
    };
    if (token) headers.Authorization = `Bearer ${token}`;
    if (options.body && !(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }
    const response = await fetch(`${baseUrl()}${path}`, {
      ...options,
      cache: "no-store",
      headers,
    });
    const text = await response.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch (_) {
      payload = text;
    }
    if (!response.ok) {
      const error = new Error(errorMessage(payload, response.status));
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  async function login(body) {
    return save(await raw("/api/v1/apps/aiops/auth/login", {
      method: "POST",
      body: JSON.stringify(body),
    }));
  }

  async function request(path, options = {}) {
    const session = load();
    if (!session?.access_token) {
      location.replace("./login.html");
      throw new Error("请先登录 AIOps");
    }
    try {
      return await raw(path, options, session.access_token);
    } catch (error) {
      if (error.status === 401) clear();
      throw error;
    }
  }

  globalThis.KBotAIOpsAuth = { clear, load, login, request, uuid };
})();
