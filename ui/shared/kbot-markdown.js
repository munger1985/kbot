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
      if (navigator.clipboard?.writeText && window.isSecureContext) {
        await navigator.clipboard.writeText(code);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = code;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        const copied = document.execCommand("copy");
        textarea.remove();
        if (!copied) throw new Error("浏览器拒绝复制操作");
      }
      button.textContent = "已复制";
    } catch (_) {
      button.textContent = "复制失败";
    }
    window.setTimeout(() => { button.textContent = original || "复制"; }, 1600);
  }

  // 只接收已经净化或由页面转义生成的 HTML，稳定更新节点而非清空整个正文。
  function patch(root, safeHtml) {
    const template = document.createElement("template");
    template.innerHTML = safeHtml;
    const key = node => node?.nodeType === 1
      ? node.getAttribute("data-model-answer") || node.getAttribute("data-turn-id") : null;
    function reconcile(current, incoming) {
      let index = 0;
      for (const next of [...incoming.childNodes]) {
        let previous = current.childNodes[index];
        const nextKey = key(next);
        if (nextKey && key(previous) !== nextKey) {
          const existing = [...current.childNodes].find(node => key(node) === nextKey);
          if (existing) {
            current.insertBefore(existing, previous || null);
            previous = existing;
          }
        }
        if (!previous) current.appendChild(next.cloneNode(true));
        else if (previous.isEqualNode(next)) { /* 保留已完成段落与选择范围。 */ }
        else if (key(previous) && key(previous) !== nextKey) {
          current.insertBefore(next.cloneNode(true), previous);
        } else if (previous.nodeType !== next.nodeType || previous.nodeName !== next.nodeName) {
          current.replaceChild(next.cloneNode(true), previous);
        } else if (previous.nodeType === Node.TEXT_NODE) {
          const before = previous.textContent || "", after = next.textContent || "";
          if (after.startsWith(before)) previous.appendData(after.slice(before.length));
          else previous.replaceData(0, before.length, after);
        } else if (previous.nodeType === Node.ELEMENT_NODE) {
          for (const attr of [...previous.attributes]) if (!next.hasAttribute(attr.name)) previous.removeAttribute(attr.name);
          for (const attr of [...next.attributes]) if (previous.getAttribute(attr.name) !== attr.value) previous.setAttribute(attr.name, attr.value);
          reconcile(previous, next);
        }
        index += 1;
      }
      while (current.childNodes.length > index) current.removeChild(current.lastChild);
    }
    reconcile(root, template.content);
  }

  globalThis.KBotMarkdown = Object.freeze({ copyCode, render, patch });
})();
