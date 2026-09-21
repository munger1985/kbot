/* 知识检索会话：登录、改密与本地 Session，不保存下游服务凭据。 */
(function () {
  "use strict";

  const storageKey = "kbot.knowledge.session.v1";

  function baseUrl() {
    const value = String(globalThis.KBOT_UI_CONFIG?.mainApiBaseUrl || "").trim().replace(/\/+$/, "");
    if (!value) throw new Error("知识检索未加载 Main API 部署配置，请刷新页面");
    return value;
  }

  function uuid() {
    return globalThis.crypto?.randomUUID?.()
      || `knowledge-ui-${Date.now()}-${Math.random().toString(16).slice(2)}`;
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
      return detail.map((item) => {
        const location = Array.isArray(item?.loc)
          ? item.loc.filter((value) => value !== "body").join(".")
          : "";
        return `${location ? `${location}：` : ""}${item?.msg || "请求内容无效"}`;
      }).join("；");
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
    if (options.body && !(options.body instanceof FormData) && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const response = await fetch(`${baseUrl()}${path}`, { cache: "no-store", ...options, headers });
    const text = await response.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch (_) { payload = text; }
    if (!response.ok) {
      const error = new Error(errorMessage(payload, response.status));
      error.status = response.status;
      error.code = payload?.code || payload?.detail?.code || "KNOWLEDGE_RETRIEVAL_AUTH_FAILED";
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  async function login(body) {
    return save(await raw("/api/v1/apps/knowledge-retrieval/auth/login", {
      method: "POST",
      body: JSON.stringify(body),
    }));
  }

  async function changePassword(body) {
    const session = load();
    if (!session?.access_token) throw new Error("请先登录知识检索");
    return save({
      ...session,
      ...(await raw("/api/v1/apps/knowledge-retrieval/auth/password", {
        method: "POST",
        body: JSON.stringify(body),
      }, session.access_token)),
    });
  }

  function passwordRuleError(value) {
    if (String(value || "").length < 12) return "新密码至少 12 位";
    if (!/[a-z]/.test(value) || !/[A-Z]/.test(value) || !/\d/.test(value) || !/[^A-Za-z0-9]/.test(value)) {
      return "新密码必须同时包含大小写字母、数字和特殊字符";
    }
    return "";
  }

  function showError(id, error) {
    const node = document.getElementById(id);
    if (!node) return;
    node.hidden = false;
    node.textContent = error?.message || "操作失败";
  }

  function reveal(id, visible) {
    const node = document.getElementById(id);
    if (node) node.hidden = !visible;
  }

  function initLoginPage() {
    const loginForm = document.getElementById("login-form");
    if (!loginForm) return;
    const passwordForm = document.getElementById("password-form");
    const session = load();
    if (session?.access_token && !session.must_change_password) {
      location.replace("./dashboard.html");
      return;
    }
    if (session?.access_token && session.must_change_password) {
      reveal("login-section", false);
      reveal("password-section", true);
    }
    loginForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      document.getElementById("login-error").hidden = true;
      const values = Object.fromEntries(new FormData(loginForm));
      try {
        const result = await login({
          user_id: String(values.user_id || "").trim(),
          password: values.password,
        });
        if (result.must_change_password) {
          reveal("login-section", false);
          reveal("password-section", true);
          return;
        }
        location.replace("./dashboard.html");
      } catch (error) {
        showError("login-error", error);
      }
    });
    passwordForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const errorNode = document.getElementById("password-error");
      if (errorNode) errorNode.hidden = true;
      const values = Object.fromEntries(new FormData(passwordForm));
      const rule = passwordRuleError(values.new_password);
      if (rule) { showError("password-error", new Error(rule)); return; }
      if (values.new_password !== values.confirm_password) {
        showError("password-error", new Error("两次输入的新密码不一致"));
        return;
      }
      try {
        await changePassword({
          current_password: values.current_password,
          new_password: values.new_password,
        });
        location.replace("./dashboard.html");
      } catch (error) {
        showError("password-error", error);
      }
    });
  }

  if (document.readyState === "loading") {
    addEventListener("DOMContentLoaded", initLoginPage, { once: true });
  } else {
    initLoginPage();
  }

  globalThis.KBotKnowledgeAuth = {
    changePassword, clear, load, login, passwordRuleError, raw, save, uuid,
  };
})();
