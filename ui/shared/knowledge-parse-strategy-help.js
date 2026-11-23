/* 两个知识库管理页共用解析方案说明。 */
(() => {
  "use strict";
  let dialog, trigger;
  document.addEventListener("click", event => {
    const button = event.target.closest("[data-knowledge-parse-help]");
    if (!button) return;
    if (!dialog) {
      dialog = document.createElement("dialog");
      dialog.className = "admin-drawer admin-drawer-sectioned reference-drawer knowledge-base-drawer";
      dialog.setAttribute("aria-labelledby", "knowledge-parse-help-title");
      dialog.innerHTML = `
        <header class="admin-drawer-header"><div><h2 id="knowledge-parse-help-title">解析方案说明</h2>
        <p>建库时不选方案；审批资料时按库模型和实际可用能力选择。</p></div>
        <button class="icon-button" type="button" data-help-close aria-label="关闭">×</button></header>
        <div class="admin-drawer-body"><div class="table-card"><table>
          <thead><tr><th scope="col">解析方案</th><th scope="col">解析效果</th><th scope="col">模型要求</th></tr></thead>
          <tbody>
            <tr><th scope="row">原生解析</th><td>保留原生文字、标题、表格和图像。扫描图像未转写的文字不可查。</td><td>不要求 OCR 或 VLM</td></tr>
            <tr><th scope="row">自动判断</th><td>原生优先；有缺字时做 OCR。可用 VLM 只补强局部图片，不做整页转写。</td><td>OCR 必需，VLM 可选</td></tr>
            <tr><th scope="row">视觉增强</th><td>在自动判断的基础上，必要时整页转写复杂版面，不覆盖可靠的原生内容。</td><td>OCR 和 VLM 都必需</td></tr>
          </tbody>
        </table></div></div>
        <footer class="admin-drawer-footer"><button class="secondary-button" type="button" data-help-close>关闭说明</button></footer>`;
      dialog.addEventListener("click", event => {
        if (event.target.closest("[data-help-close]")) dialog.close();
      });
      dialog.addEventListener("close", () => {
        trigger?.setAttribute("aria-expanded", "false");
        trigger?.focus();
        trigger = null;
      });
      document.body.append(dialog);
    }
    if (dialog.open) return;
    trigger = button;
    button.setAttribute("aria-expanded", "true");
    dialog.showModal();
  });
})();
