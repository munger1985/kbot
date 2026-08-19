/* 与 Ammolite AgentMarkdownContent 同源：Marked GFM + DOMPurify 安全清洗。 */
(function () {
  "use strict";

  const markedApi = globalThis.marked;
  const purifier = globalThis.DOMPurify;

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[character]);
  }

  function requireRuntime() {
    if (!markedApi?.Marked || !markedApi?.Renderer || !purifier?.sanitize) {
      throw new Error("完整 Markdown 渲染依赖未加载");
    }
  }

  requireRuntime();

  const renderer = new markedApi.Renderer();
  renderer.code = ({ text, lang }) => {
    const language = String(lang || "plaintext")
      .trim().split(/\s+/, 1)[0].toLowerCase()
      .replace(/[^a-z0-9_+-]/g, "") || "plaintext";
    return `<div class="agent-code-block"><div class="agent-code-toolbar"><span>${escapeHtml(language.toUpperCase())}</span><button type="button" class="row-action agent-code-copy" data-copy-code>复制</button></div><pre><code class="language-${escapeHtml(language)}">${escapeHtml(text)}</code></pre></div>`;
  };
  renderer.link = function ({ href, title, tokens }) {
    const label = this.parser.parseInline(tokens);
    const safeHref = /^(https?:|mailto:|\/)/i.test(href) ? href : "#";
    const safeTitle = title ? ` title="${escapeHtml(title)}"` : "";
    return `<a href="${escapeHtml(safeHref)}"${safeTitle} target="_blank" rel="noopener noreferrer">${label}</a>`;
  };

  const markdown = new markedApi.Marked({
    gfm: true,
    breaks: true,
    async: false,
    renderer,
  });

  function render(value) {
    const source = String(value ?? "");
    try {
      const raw = String(markdown.parse(source));
      return purifier.sanitize(raw, {
        USE_PROFILES: { html: true },
        ADD_ATTR: ["class", "target", "rel", "data-copy-code", "start"],
        FORBID_TAGS: ["style", "script", "iframe", "object", "embed", "form", "input"],
        FORBID_ATTR: ["style", "srcset"],
        ALLOW_DATA_ATTR: false,
      });
    } catch (_) {
      return purifier.sanitize(escapeHtml(source).replaceAll("\n", "<br>"));
    }
  }

  async function copyCode(button) {
    const code = button?.closest?.(".agent-code-block")?.querySelector("code")?.textContent || "";
    if (!code) return;
    const original = button.textContent;
    try {
      await navigator.clipboard.writeText(code);
      button.textContent = "已复制";
    } catch (_) {
      button.textContent = "复制失败";
    }
    window.setTimeout(() => { button.textContent = original || "复制"; }, 1600);
  }

  globalThis.KBotMarkdown = Object.freeze({ copyCode, render });
})();
