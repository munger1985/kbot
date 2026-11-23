/* 读取统一工作台签发的业务会话，不保存登录凭据。 */
(() => {
  "use strict";
  const key = "kbot.knowledge.session.v1";
  const uuid = () => globalThis.crypto?.randomUUID?.()
    || `knowledge-ui-${Date.now()}-${Math.random().toString(16).slice(2)}`;

  function load() {
    try {
      return JSON.parse(sessionStorage.getItem(key) || "null");
    } catch (_) {
      return null;
    }
  }

  function save(session) {
    sessionStorage.setItem(key, JSON.stringify(session));
    return session;
  }

  const clear = () => sessionStorage.removeItem(key);
  globalThis.KBotKnowledgeAuth = { clear, load, save, uuid };
})();
