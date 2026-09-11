/* X 搜索测试页：保留输入校验，但不在没有已验收模型时创建虚假 Run。 */
(function () {
  "use strict";
  addEventListener("DOMContentLoaded", () => {
    document.getElementById("x-search-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const from = document.getElementById("x-search-from").value;
      const to = document.getElementById("x-search-to").value;
      if (from && to && from > to) { KBotAssistantShell.toast("起始日期不能晚于结束日期。", "error"); return; }
      KBotAssistantShell.toast("模型能力尚未验收，未创建 X Search Run。", "error");
    });
    ["x-search-new", "x-search-refresh"].forEach((id) => document.getElementById(id).addEventListener("click", () => KBotAssistantShell.toast("等待 X Search Run API 接入。")));
  }, { once: true });
})();
