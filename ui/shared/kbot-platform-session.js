/* KBot 统一会话：业务 Token 与根会话分开保存，不保存登录凭据。 */
(() => {
  "use strict";

  const key = "kbot.platform.session.v1";
  const rootKinds = new Set(["PLATFORM", "WORKSPACE"]);
  const notificationKey = "kbot.aiops.notification_entry.v1";
  const apps = Object.freeze({
    knowledge_retrieval: { key: "kbot.knowledge.session.v1", home: "/ui/knowledge_retrieval/knowledge.html" },
    aiops: { key: "kbot.aiops.session.v1", home: "/ui/aiops/dashboard.html" },
    media_studio: { key: "kbot.media.session.v1", home: "/ui/media_studio/dashboard.html" },
  });
  let renewal = null;
  let sessionGeneration = 0;

  function load() {
    try {
      const session = JSON.parse(sessionStorage.getItem(key) || "null");
      return rootKinds.has(session?.entry_kind) && session.access_token ? session : null;
    } catch (_) {
      return null;
    }
  }

  function save(session) {
    if (!rootKinds.has(session?.entry_kind) || !session.access_token) throw new Error("未收到工作台会话");
    sessionStorage.setItem(key, JSON.stringify(session));
    return session;
  }

  function clear({ preserveNotification = false } = {}) {
    sessionGeneration += 1;
    sessionStorage.removeItem(key);
    for (const app of Object.values(apps)) sessionStorage.removeItem(app.key);
    if (!preserveNotification) sessionStorage.removeItem(notificationKey);
  }

  function pendingNotification() {
    const id = sessionStorage.getItem(notificationKey) || "";
    return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(id) ? id : null;
  }

  function logout() {
    clear();
    location.replace("/ui/index.html");
  }

  async function send(path, method, body, session) {
    const base = String(globalThis.KBOT_UI_CONFIG?.mainApiBaseUrl || "").trim().replace(/\/+$/, "");
    if (!base) throw new Error("未加载 Main API 部署配置，请刷新页面");
    const headers = {
      Accept: "application/json",
      "X-Request-ID": globalThis.crypto?.randomUUID?.() || `platform-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    };
    if (session) headers.Authorization = `Bearer ${session.access_token}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const response = await fetch(`${base}/api/v1${path}`, {
      method, headers, cache: "no-store",
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail;
      const fields = payload?.field_errors || (Array.isArray(detail) ? detail : []);
      const fieldMessage = fields.map((item) => {
        const path = (item.location || item.loc || []).filter((part) => part !== "body").join(".");
        return [path, item.message || item.msg].filter(Boolean).join("：");
      }).join("；");
      const error = new Error(
        fieldMessage || (typeof detail === "string" ? detail : detail?.message)
        || payload?.message || `请求失败（HTTP ${response.status}）`,
      );
      error.status = response.status;
      error.code = payload?.code || detail?.code;
      throw error;
    }
    return payload;
  }

  async function login(body) {
    const session = await send("/auth/login", "POST", body);
    clear({ preserveNotification: true });
    return save(session);
  }

  async function request(path, method = "GET", body) {
    let session = load();
    if (!session) throw Object.assign(new Error("请先登录 KBot"), { status: 401 });
    const generation = sessionGeneration;
    try {
      const remaining = Date.parse(session.expires_at) - Date.now();
      if (!session.must_change_password && path !== "/auth/password" && remaining > 0 && remaining < 120_000) {
        if (!renewal) {
          const current = session;
          renewal = send("/auth/refresh", "POST", undefined, current).then((next) => {
            if (load()?.access_token === current.access_token) return save(next);
            throw Object.assign(new Error("工作台会话已切换，请重试"), { status: 401 });
          }).finally(() => { renewal = null; });
        }
        session = await renewal;
      }
      const result = await send(path, method, body, session);
      if (generation !== sessionGeneration) {
        throw Object.assign(new Error("工作台会话已切换，请重试"), { status: 401 });
      }
      return result;
    } catch (error) {
      const passwordRejected = path === "/auth/password" && ["INVALID_CREDENTIALS", "PASSWORD_REUSED"].includes(error.code);
      if (error.status === 401 && error.code !== "PASSWORD_CHANGE_REQUIRED" && !passwordRejected
          && generation === sessionGeneration
          && load()?.access_token === session.access_token) clear({ preserveNotification: true });
      throw error;
    }
  }

  async function changePassword(body) {
    const next = await request("/auth/password", "POST", body);
    for (const app of Object.values(apps)) sessionStorage.removeItem(app.key);
    return save(next);
  }

  function openApp(appId, session, domainId) {
    const app = Object.hasOwn(apps, appId) ? apps[appId] : null;
    if (!app) throw new Error("未知应用入口");
    if (session?.entry_kind !== "BUSINESS" || session.app_id !== appId
        || !Number.isSafeInteger(session.domain_id) || session.domain_id <= 0
        || (domainId !== undefined && session.domain_id !== domainId) || session.user_id !== load()?.user_id
        || !session.access_token || session.must_change_password) {
      throw new Error("未收到匹配的应用会话");
    }
    sessionStorage.setItem(app.key, JSON.stringify(session));
    const notification = appId === "aiops" ? pendingNotification() : null;
    location.assign(notification
      ? `/ui/aiops/notification-open.html?entry=${encodeURIComponent(notification)}` : app.home);
  }

  async function enterApp(appId, domainId) {
    const session = await request("/auth/exchange", "POST", { app_id: appId, domain_id: domainId });
    openApp(appId, session, domainId);
  }

  async function createKnowledgeFirstDomain(body) {
    const session = await request("/auth/knowledge-first-domain", "POST", body);
    openApp("knowledge_retrieval", session);
  }

  globalThis.KBotPlatformSession = {
    load, clear, login, logout, request, changePassword, enterApp, createKnowledgeFirstDomain, pendingNotification,
  };
})();
