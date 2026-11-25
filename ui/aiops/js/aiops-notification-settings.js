import { mountFeishuSettings } from "../../shared/kbot-feishu-settings.js?v=20261125_self_feishu";

KBotAIOpsShell.ready.then((access) => {
  if (!access) return;
  const settings = mountFeishuSettings(document.getElementById("delivery-mine"), (path, method, body) =>
    KBotAIOpsAuth.request("/api/v1" + path, body === undefined ? {} : { method, body: JSON.stringify(body) }));
  document.getElementById("delivery-refresh").onclick = settings.refresh;
  void settings.refresh();
});
