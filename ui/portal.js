(() => {
  "use strict";
  const session = KBotPlatformSession;
  const el = (id) => document.getElementById(id);
  const enabled = globalThis.KBOT_UI_CONFIG?.enabledApps;
  const isEnabled = (id) => !Array.isArray(enabled) || enabled.includes(id);
  const cards = Array.from(document.querySelectorAll("[data-app]"));
  const login = el("portal-login");
  const password = el("portal-password");
  const status = el("portal-status");
  const retry = el("portal-retry");
  const dialog = el("portal-domain-dialog");
  const domainForm = el("portal-domain-form");
  const firstDomainDialog = el("portal-first-domain-dialog");
  const firstDomainForm = el("portal-first-domain-form");
  const firstDomainError = el("portal-first-domain-error");
  let entries = new Map();
  let selectedApp = null;
  let pending = false;
  let discoveryGeneration = 0;

  function showError(message, node = el("portal-error")) {
    node.textContent = message || "";
    node.hidden = !message;
  }

  function showScreen() {
    const user = session.load();
    document.body.dataset.portalView = user && !user.must_change_password ? "workspace" : "auth";
    el("portal-account").hidden = !user;
    el("portal-user").textContent = user?.display_name || user?.user_id || "";
    el("portal-sign-in").hidden = Boolean(user);
    password.hidden = !user?.must_change_password;
    el("portal-title").textContent = !user ? "登录工作台" : user.must_change_password ? "修改初始密码" : "选择工作台";
    el("portal-description").textContent = !user ? "请使用你的 KBot 账号。"
      : user.must_change_password ? "修改初始密码后才能进入工作台。" : "选择已授权的应用或系统管理入口。";
    return user;
  }

  async function discover() {
    const generation = ++discoveryGeneration;
    entries = new Map();
    for (const card of cards) card.hidden = true;
    el("system-entry").hidden = true;
    el("empty-apps").hidden = true;
    el("portal-notification").hidden = true;
    retry.hidden = true;
    status.hidden = true;
    showError("");
    const user = showScreen();
    if (!user || user.must_change_password) return;
    status.textContent = "正在读取已授权的工作台…";
    status.hidden = false;
    try {
      const result = await session.request("/auth/entries");
      if (generation !== discoveryGeneration) return;
      for (const entry of result.apps) {
        const domains = entry.domains.filter((domain) => domain.status === "ACTIVE"
          && Number.isSafeInteger(domain.domain_id) && domain.domain_id > 0);
        const canCreate = entry.app_id === "knowledge_retrieval" && entry.can_create_first_domain === true;
        if (isEnabled(entry.app_id) && (domains.length || canCreate)) {
          entries.set(entry.app_id, { ...entry, domains, canCreate });
        }
      }
      for (const card of cards) card.hidden = !entries.has(card.dataset.app);
      el("system-entry").hidden = result.system_management !== true;
      el("empty-apps").hidden = !el("system-entry").hidden || cards.some((card) => !card.hidden);
      el("portal-notification").hidden = !entries.has("aiops") || !session.pendingNotification();
    } catch (error) {
      if (generation !== discoveryGeneration) return;
      showScreen();
      retry.hidden = !session.load();
      showError(session.load() ? error.message : "登录已失效，请重新登录。");
    } finally {
      if (generation === discoveryGeneration) status.hidden = true;
    }
  }

  async function run(form, operation) {
    if (pending) return;
    pending = true;
    const submit = form?.querySelector('[type="submit"]');
    if (submit) submit.disabled = true;
    showError("");
    showError("", firstDomainError);
    try {
      await operation();
    } catch (error) {
      if (dialog.open) dialog.close();
      if (!session.load() && login !== form) {
        if (firstDomainDialog.open) firstDomainDialog.close();
        await discover();
      }
      if (error.code === "FIRST_DOMAIN_ALREADY_EXISTS" && firstDomainDialog.open) {
        firstDomainDialog.close();
        await discover();
        showError("首个工作域已创建。请选择你有权限进入的工作域。");
      } else {
        showError(error.message || "操作失败，请重试。", firstDomainDialog.open ? firstDomainError : undefined);
      }
    } finally {
      if (submit) submit.disabled = false;
      pending = false;
    }
  }

  login.addEventListener("submit", (event) => {
    event.preventDefault();
    run(login, async () => {
      const values = Object.fromEntries(new FormData(login));
      try {
        await session.login({ user_id: values.user_id.trim(), password: values.password });
      } finally {
        login.elements.password.value = "";
      }
      await discover();
    });
  });
  password.addEventListener("submit", (event) => {
    event.preventDefault();
    run(password, async () => {
      try {
        await session.changePassword(Object.fromEntries(new FormData(password)));
      } finally {
        password.reset();
      }
      await discover();
    });
  });

  for (const card of cards) {
    card.addEventListener("click", () => {
      const entry = entries.get(card.dataset.app);
      if (!entry || pending) return;
      if (entry.canCreate && entry.domains.length === 0) {
        firstDomainForm.reset();
        showError("", firstDomainError);
        firstDomainDialog.showModal();
        return;
      }
      if (entry.domains.length === 1) {
        run(null, () => session.enterApp(entry.app_id, entry.domains[0].domain_id));
        return;
      }
      selectedApp = entry.app_id;
      el("portal-domain-title").textContent = `选择工作域 · ${entry.display_name}`;
      domainForm.elements.domain_id.replaceChildren(...entry.domains.map(
        (domain) => new Option(domain.name, String(domain.domain_id)),
      ));
      dialog.showModal();
    });
  }
  el("portal-domain-cancel").onclick = () => { if (!pending) dialog.close(); };
  dialog.addEventListener("cancel", (event) => { if (pending) event.preventDefault(); });
  domainForm.addEventListener("submit", (event) => {
    event.preventDefault();
    run(domainForm, () => session.enterApp(selectedApp, Number(domainForm.elements.domain_id.value)));
  });
  el("portal-first-domain-cancel").onclick = () => { if (!pending) firstDomainDialog.close(); };
  firstDomainDialog.addEventListener("cancel", (event) => { if (pending) event.preventDefault(); });
  firstDomainForm.addEventListener("submit", (event) => {
    event.preventDefault();
    run(firstDomainForm, () => {
      const { name, description } = Object.fromEntries(new FormData(firstDomainForm));
      return session.createKnowledgeFirstDomain({ name: name.trim(), description: description.trim() || null });
    });
  });
  el("portal-logout").onclick = () => {
    session.clear();
    if (dialog.open) dialog.close();
    if (firstDomainDialog.open) firstDomainDialog.close();
    discover();
  };
  retry.onclick = discover;
  addEventListener("pageshow", (event) => { if (event.persisted) discover(); });
  discover();
})();
