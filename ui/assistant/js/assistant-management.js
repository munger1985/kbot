/* 管理测试页只校验交互路径；筛选动作等待对应 API 提供受权限投影的结果。 */
(function () {
  "use strict";
  addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("#asset-filter, #runs-filter, #runs-refresh").forEach((button) => button.addEventListener("click", () => KBotAssistantShell.toast("管理 API 尚未接入，未加载或修改任何记录。")));
  }, { once: true });
})();
