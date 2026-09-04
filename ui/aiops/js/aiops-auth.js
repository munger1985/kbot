(function () {
  "use strict";

  const storageKey = "kbot.aiops.session.v1";
  function baseUrl() {
    const value = String(
      globalThis.KBOT_UI_CONFIG?.mainApiBaseUrl || "",
    ).trim().replace(/\/+$/, "");
    if (!value) {
      throw new Error("AIOps UI 未加载 Main API 部署配置，请刷新页面");
    }
    return value;
  }

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

  async function requestBlob(path, options = {}) {
    const session = load();
    if (!session?.access_token) {
      location.replace("./login.html");
      throw new Error("请先登录 AIOps");
    }
    const response = await fetch(`${baseUrl()}${path}`, {
      ...options,
      cache: "no-store",
      headers: {
        Accept: "image/*",
        "X-Request-ID": uuid(),
        Authorization: `Bearer ${session.access_token}`,
        ...(options.headers || {}),
      },
    });
    if (!response.ok) {
      const text = await response.text();
      let payload = text;
      try { payload = text ? JSON.parse(text) : null; } catch (_) { /* 保留原始错误文本。 */ }
      if (response.status === 401) clear();
      throw new Error(errorMessage(payload, response.status));
    }
    return response.blob();
  }

  async function stream(path, onEvent, options = {}) {
    const session = load();
    if (!session?.access_token) {
      location.replace("./login.html");
      throw new Error("请先登录 AIOps");
    }
    const response = await fetch(`${baseUrl()}${path}`, {
      ...options,
      cache: "no-store",
      headers: {
        Accept: "text/event-stream",
        Authorization: `Bearer ${session.access_token}`,
        "X-Request-ID": uuid(),
        ...(options.headers || {}),
      },
    });
    if (!response.ok || !response.body) {
      const payload = await response.text();
      throw new Error(payload || `事件流连接失败（HTTP ${response.status}）`);
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
        let event = "message";
        let id = "";
        let data = "";
        for (const line of block.split(/\r?\n/)) {
          if (line.startsWith("id:")) id = line.slice(3).trim();
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        let payload = data;
        try { payload = data ? JSON.parse(data) : null; } catch (_) { /* 保留文本事件。 */ }
        await onEvent({ event, data: payload, id });
      }
      if (done) break;
    }
  }

  async function download(path, fileName) {
    const session = load();
    if (!session?.access_token) {
      location.replace("./login.html");
      throw new Error("请先登录 AIOps");
    }
    const response = await fetch(`${baseUrl()}${path}`, {
      cache: "no-store",
      headers: {
        Accept: "application/pdf",
        Authorization: `Bearer ${session.access_token}`,
        "X-Request-ID": uuid(),
      },
    });
    if (!response.ok) {
      const payload = await response.text();
      let parsed = payload;
      try { parsed = payload ? JSON.parse(payload) : null; } catch (_) { /* 保留原始错误文本。 */ }
      throw new Error(errorMessage(parsed, response.status));
    }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  globalThis.KBotAIOpsAuth = { clear, load, login, request, requestBlob, stream, download, uuid };
})();
