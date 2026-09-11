/* 资源配置测试页的 Dialog 与表单骨架；后端接入前不在浏览器虚构资源状态。 */
(function () {
  "use strict";
  function openDialog(id) { document.getElementById(id)?.showModal(); }
  function closeDialog(id) { document.getElementById(id)?.close(); }
  addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-open-dialog]").forEach((button) => button.addEventListener("click", () => openDialog(button.dataset.openDialog)));
    document.querySelectorAll("[data-close-dialog]").forEach((button) => button.addEventListener("click", () => closeDialog(button.dataset.closeDialog)));
    document.querySelectorAll("form[data-resource-form]").forEach((form) => form.addEventListener("submit", (event) => {
      event.preventDefault();
      KBotAssistantShell.toast(`${form.dataset.resourceForm} API 尚未接入，未创建资源。`, "error");
    }));
  }, { once: true });
  globalThis.KBotAssistantResources = { closeDialog, openDialog };
})();
