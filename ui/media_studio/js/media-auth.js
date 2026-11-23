/* 读取统一工作台签发的业务会话，不保存登录凭据。 */
(() => {
  "use strict";
  const key = "kbot.media.session.v1";
  const uuid = () => globalThis.crypto?.randomUUID?.()
    || `media-ui-${Date.now()}-${Math.random().toString(16).slice(2)}`;

  function load() {
    try {
      return JSON.parse(sessionStorage.getItem(key) || "null");
    } catch (_) {
      return null;
    }
  }

  const clear = () => sessionStorage.removeItem(key);
  globalThis.KBotMediaAuth = { clear, load, uuid };
})();
