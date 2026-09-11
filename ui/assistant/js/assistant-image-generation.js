/* 文生图测试页：在模型与配额能力接入前不伪造图像资产。 */
(function () {
  "use strict";
  addEventListener("DOMContentLoaded", () => {
    document.getElementById("image-form").addEventListener("submit", (event) => { event.preventDefault(); KBotAssistantShell.toast("模型能力尚未验收，未创建图片生成任务。", "error"); });
    ["image-new", "image-refresh"].forEach((id) => document.getElementById(id).addEventListener("click", () => KBotAssistantShell.toast("等待图片生成与资产 API 接入。")));
  }, { once: true });
})();
