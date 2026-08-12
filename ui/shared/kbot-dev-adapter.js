/* 仅在后端 8080 开发 UI 服务中启用测试认证；生产 Portal 不加载测试身份。 */
(function () {
  "use strict";
  if (window.location.port !== "8080") return;

  const storageKey = "kbot.km.development.connection.v1";
  function load() {
    let value = {};
    try { value = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) { value = {}; }
    return {
      baseUrl: value.baseUrl || `${location.protocol}//${location.hostname}:18099`,
      domainId: String(value.domainId || ""),
      userId: String(value.userId || "kbotui_dev"),
    };
  }
  function save(value) {
    localStorage.setItem(storageKey, JSON.stringify({
      baseUrl: String(value.baseUrl || "").replace(/\/+$/, ""),
      domainId: String(value.domainId || "").trim(),
      userId: String(value.userId || "").trim(),
    }));
    return load();
  }
  function headers(extra) {
    const config = load();
    return {
      "X-KBot-Test-Auth": "true",
      "X-KBot-Domain-ID": config.domainId,
      "X-KBot-User-ID": config.userId,
      "X-Request-ID": KBotKmApi.uuid(),
      ...(extra || {}),
    };
  }
  function requireConfig() {
    const config = load();
    if (!config.domainId || !config.userId) throw new Error("请先设置 Domain ID 和 User ID");
    return config;
  }
  async function decode(response) {
    const text = await response.text();
    if (!text) return null;
    try { return JSON.parse(text); } catch (_) { return text; }
  }
  function error(response, payload) {
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : detail?.message || payload?.message || payload?.title || `请求失败（HTTP ${response.status}）`;
    const failure = new Error(message);
    failure.status = response.status;
    failure.requestId = payload?.request_id || response.headers.get("X-Request-ID") || "";
    return failure;
  }
  async function request(path, options = {}) {
    const config = requireConfig();
    const requestHeaders = headers(options.headers);
    if (options.body && !(options.body instanceof FormData) && !requestHeaders["Content-Type"]) requestHeaders["Content-Type"] = "application/json";
    const response = await fetch(`${config.baseUrl}${path}`, { ...options, headers: requestHeaders });
    const payload = await decode(response);
    if (!response.ok) throw error(response, payload);
    return payload;
  }
  async function blob(path, options = {}) {
    const config = requireConfig();
    const response = await fetch(`${config.baseUrl}${path}`, { ...options, headers: headers(options.headers) });
    if (!response.ok) throw error(response, await decode(response));
    return { data: await response.blob(), contentType: response.headers.get("Content-Type") || "application/octet-stream" };
  }
  async function stream(path, handlers = {}, signal) {
    const config = requireConfig();
    const response = await fetch(`${config.baseUrl}${path}`, { headers: headers({ Accept: "text/event-stream", "Last-Event-ID": String(handlers.lastEventId || 0) }), signal });
    if (!response.ok || !response.body) throw error(response, await decode(response));
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
  window.KBotKmDev = { active: true, load, save };
})();
