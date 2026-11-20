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
    return raw("/api/v1/apps/knowledge-retrieval/auth/login", {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  async function completeEntry(body, ticket) {
    return raw("/api/v1/apps/knowledge-retrieval/auth/entry", {
      method: "POST",
      body: JSON.stringify(body),
    }, ticket);
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
    const selectForm = document.getElementById("select-domain-form");
    const firstDomainForm = document.getElementById("first-domain-form");
    let entry = null;
    let pending = false;

    function showStep(step) {
      for (const section of ["login", "entry", "password"]) {
        reveal(`${section}-section`, section === step);
      }
    }

    function enter(result) {
      if (result?.action !== "ENTERED" || !result.access_token || !result.domain_id) {
        throw new Error("未收到有效的知识检索业务会话");
      }
      entry = null;
      save(result);
      if (result.must_change_password) {
        showStep("password");
        passwordForm?.querySelector("input")?.focus();
      } else {
        location.replace("./knowledge.html");
      }
    }

    function requestDomain(result) {
      if (!result.entry_ticket || !["SELECT_DOMAIN", "CHANGE_PASSWORD", "CREATE_FIRST_DOMAIN"].includes(result.action)) {
        throw new Error("未收到有效的工作域入口");
      }
      entry = { action: result.action, ticket: result.entry_ticket };
      if (result.action === "CHANGE_PASSWORD") {
        showStep("password");
        passwordForm?.querySelector("input")?.focus();
        return;
      }
      document.getElementById("entry-error").hidden = true;
      const select = result.action === "SELECT_DOMAIN";
      if (select) {
        const options = result.domains.map(row => new Option(row.name, String(row.domain_id)));
        selectForm.elements.domain_id.replaceChildren(...options);
      }
      reveal("select-domain-form", select);
      reveal("first-domain-form", !select);
      showStep("entry");
      (select ? selectForm.elements.domain_id : firstDomainForm.elements.name).focus();
    }

    const session = load();
    if (session?.access_token && !session.must_change_password) {
      location.replace("./knowledge.html");
      return;
    }
    if (session?.access_token && session.must_change_password) {
      showStep("password");
    }
    loginForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (pending) return;
      document.getElementById("login-error").hidden = true;
      const values = Object.fromEntries(new FormData(loginForm));
      pending = true;
      const submit = loginForm.querySelector('button[type="submit"]');
      submit.disabled = true;
      try {
        const result = await login({
          user_id: String(values.user_id || "").trim(),
          password: values.password,
        });
        loginForm.elements.password.value = "";
        if (result.action === "ENTERED") enter(result);
        else requestDomain(result);
      } catch (error) {
        showError("login-error", error);
      } finally {
        submit.disabled = false;
        pending = false;
      }
    });
    document.getElementById("entry-back")?.addEventListener("click", () => {
      if (pending) return;
      entry = null;
      firstDomainForm.reset();
      showStep("login");
      loginForm.elements.password.focus();
    });
    for (const form of [selectForm, firstDomainForm]) {
      form?.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (pending || !entry) return;
        document.getElementById("entry-error").hidden = true;
        const values = new FormData(form);
        const body = entry.action === "SELECT_DOMAIN"
          ? { action: entry.action, domain_id: Number(values.get("domain_id")) }
          : { action: entry.action, name: String(values.get("name")).trim(),
              description: String(values.get("description") || "").trim() || null };
        const submit = form.querySelector('button[type="submit"]');
        submit.disabled = true;
        pending = true;
        try {
          enter(await completeEntry(body, entry.ticket));
        } catch (error) {
          if (error.status === 401 || error.code === "FIRST_DOMAIN_ALREADY_EXISTS") {
            entry = null;
            showStep("login");
            showError("login-error", error);
          } else {
            showError("entry-error", error);
          }
        } finally {
          submit.disabled = false;
          pending = false;
        }
      });
    }
    passwordForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (pending) return;
      const errorNode = document.getElementById("password-error");
      if (errorNode) errorNode.hidden = true;
      const values = Object.fromEntries(new FormData(passwordForm));
      const rule = passwordRuleError(values.new_password);
      if (rule) { showError("password-error", new Error(rule)); return; }
      if (values.new_password !== values.confirm_password) {
        showError("password-error", new Error("两次输入的新密码不一致"));
        return;
      }
      const body = { current_password: values.current_password, new_password: values.new_password };
      const submit = passwordForm.querySelector('button[type="submit"]');
      submit.disabled = true;
      pending = true;
      try {
        if (entry?.action === "CHANGE_PASSWORD") {
          const next = await completeEntry({ action: entry.action, ...body }, entry.ticket);
          passwordForm.reset();
          requestDomain(next);
        } else {
          await changePassword(body);
          location.replace("./knowledge.html");
        }
      } catch (error) {
        if (entry && ["ENTRY_TICKET_EXPIRED", "INVALID_ENTRY_TICKET", "USER_SESSION_REVOKED",
          "FIRST_DOMAIN_ALREADY_EXISTS"].includes(error.code)) {
          entry = null;
          passwordForm.reset();
          showStep("login");
          showError("login-error", error);
        } else {
          showError("password-error", error);
        }
      } finally {
        submit.disabled = false;
        pending = false;
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
