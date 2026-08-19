/* KBot development 日志页面共享工具。 */
(function () {
  "use strict";

  function mainApiBaseUrl() {
    const hostname = window.location.hostname;
    if (hostname && hostname !== "127.0.0.1" && hostname !== "localhost") {
      return `${window.location.protocol}//${hostname}:18099`;
    }
    return "http://127.0.0.1:18099";
  }

  function uuid() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(
      /[xy]/g,
      (character) => {
        const value = Math.floor(Math.random() * 16);
        const nibble = character === "x" ? value : (value & 0x3) | 0x8;
        return nibble.toString(16);
      }
    );
  }

  async function developmentLogApi(path, options) {
    const requestOptions = { ...(options || {}) };
    const headers = {
      "X-KBot-Test-Auth": "true",
      "X-KBot-User-ID": "development-log-console",
      "X-Request-ID": uuid(),
      ...(requestOptions.headers || {}),
    };
    const response = await fetch(`${mainApiBaseUrl()}${path}`, {
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
      throw new Error(
        `HTTP ${response.status}: ${
          typeof detail === "string" ? detail : JSON.stringify(detail)
        }`
      );
    }
    return payload;
  }

  function setStatus(element, message, kind) {
    element.textContent = message || "";
    element.className = `status${kind ? ` ${kind}` : ""}`;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  window.KBotUI = {
    developmentLogApi,
    escapeHtml,
    setStatus,
  };
})();
