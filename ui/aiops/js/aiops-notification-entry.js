(async function () {
  "use strict";
  const entryId = new URLSearchParams(location.search).get("entry") || "";
  const valid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(entryId);
  const message = document.getElementById("entry-status");
  function login() { sessionStorage.setItem("kbot.aiops.notification_entry.v1", entryId); location.replace("./login.html"); }
  if (!valid) { message.textContent = "通知入口格式无效"; return; }
  if (!KBotAIOpsAuth.load()?.access_token) { login(); return; }
  try {
    const entry = await KBotAIOpsAuth.request("/api/v1/notification-delivery/entries/" + encodeURIComponent(entryId));
    const access = await KBotAIOpsAuth.request("/api/v1/apps/aiops/access");
    if (String(access.domain_id) !== entry.domain_id) throw new Error("当前登录空间与通知空间不同，请使用对应空间登录；不会跳转其他任务。");
    const destination = KBotNotificationCenter.href({...entry, domain_id: entry.domain_id}, entry.domain_id);
    if (!destination) throw new Error("通知原任务定位不可用");
    sessionStorage.removeItem("kbot.aiops.notification_entry.v1");
    location.replace(destination);
  } catch (error) {
    if (error.status === 401) { login(); return; }
    message.textContent = error.message || "入口已过期、资源不可用或当前账号无权访问";
  }
})();
