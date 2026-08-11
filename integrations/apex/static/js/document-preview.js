(function (global) {
  "use strict";

  let contentLoader = null;

  const encode = (value) => encodeURIComponent(String(value || ""));

  function api(path) {
    return global.KBotApi.request(path, { label: "文档预览" });
  }

  function previewPath(runId, citationLabel) {
    return `/api/v1/apps/knowledge-retrieval/runs/${encode(runId)}`
      + `/references/${encode(citationLabel)}/preview`;
  }

  async function defaultLoadContent(url) {
    const response = await global.fetch(url, {
      method: "GET",
      credentials: "same-origin",
      headers: { Accept: "*/*" }
    });
    if (!response.ok) {
      const error = new Error(`文档内容加载失败（HTTP ${response.status}）`);
      error.status = response.status;
      throw error;
    }
    return response.blob();
  }

  function showError(error) {
    const message = error?.message || "文档预览失败。";
    if (global.apex?.message?.showErrors) {
      global.apex.message.clearErrors();
      global.apex.message.showErrors([{
        type: "error",
        location: "page",
        message,
        unsafe: false
      }]);
      return;
    }
    global.alert(message);
  }

  function reserveViewer() {
    const viewer = global.open("about:blank", "_blank");
    if (!viewer) {
      throw new Error("浏览器阻止了文档预览窗口，请允许本站打开新窗口。");
    }
    viewer.opener = null;
    viewer.document.title = "正在加载来源文档";
    viewer.document.body.textContent = "正在加载来源文档…";
    return viewer;
  }

  function openBlob(blob, descriptor, viewer) {
    const objectUrl = URL.createObjectURL(blob);
    if (descriptor.preview_type === "DOWNLOAD") {
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = descriptor.title || "document";
      link.click();
      viewer.close();
      global.setTimeout(() => URL.revokeObjectURL(objectUrl), 30 * 1000);
      return;
    }
    const page = descriptor.preview_type === "PDF" && descriptor.page_no
      ? `#page=${descriptor.page_no}`
      : "";
    viewer.location.replace(`${objectUrl}${page}`);
    global.setTimeout(() => URL.revokeObjectURL(objectUrl), 5 * 60 * 1000);
  }

  async function open(options) {
    const runId = options?.runId;
    const citationLabel = options?.citationLabel;
    if (!runId || !citationLabel) {
      throw new Error("文档预览缺少 runId 或 citationLabel。");
    }
    const viewer = reserveViewer();
    try {
      const descriptor = await api(previewPath(runId, citationLabel));
      const loader = options.loadContent || contentLoader || defaultLoadContent;
      const blob = await loader(descriptor.content_url, descriptor);
      if (!(blob instanceof Blob)) {
        throw new Error("文档内容加载器必须返回 Blob。");
      }
      openBlob(blob, descriptor, viewer);
      return descriptor;
    } catch (error) {
      viewer.close();
      showError(error);
      throw error;
    }
  }

  function configure(options) {
    if (options?.loadContent && typeof options.loadContent !== "function") {
      throw new TypeError("loadContent 必须是函数。");
    }
    contentLoader = options?.loadContent || null;
  }

  function bind(container, options = {}) {
    const root = typeof container === "string"
      ? document.querySelector(container)
      : container;
    if (!root) return;
    root.addEventListener("click", (event) => {
      const trigger = event.target.closest("[data-kbot-citation-label]");
      if (!trigger || !root.contains(trigger)) return;
      event.preventDefault();
      open({
        runId: trigger.dataset.kbotRunId || options.runId,
        citationLabel: trigger.dataset.kbotCitationLabel,
        loadContent: options.loadContent
      }).catch(() => {});
    });
  }

  global.KBotDocumentPreview = Object.freeze({ bind, configure, open });
})(window);
