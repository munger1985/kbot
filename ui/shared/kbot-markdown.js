/* KBot 页面共用的安全 Markdown 子集渲染器，不执行原始 HTML。 */
(function () {
  "use strict";

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[character]);
  }

  function inline(value) {
    const fragments = [];
    const hold = (html) => `\u0000${fragments.push(html) - 1}\u0000`;
    let source = String(value ?? "");
    source = source.replace(/`([^`\n]+)`/g, (_, code) => hold(`<code>${escapeHtml(code)}</code>`));
    source = source.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/gi, (_, label, url) => hold(
      `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(label)}</a>`,
    ));
    source = escapeHtml(source)
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
      .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
      .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
      .replace(/\[Q(\d+)\]/g, '<span class="km-query-citation">[Q$1]</span>');
    return source.replace(/\u0000(\d+)\u0000/g, (_, index) => fragments[Number(index)] || "");
  }

  function render(value) {
    const lines = String(value ?? "").replace(/\r\n?/g, "\n").split("\n");
    const output = [];
    let paragraph = [];
    let listType = "";
    let codeFence = false;
    let codeLanguage = "";
    let codeLines = [];
    const closeParagraph = () => {
      if (paragraph.length) output.push(`<p>${paragraph.map(inline).join("<br>")}</p>`);
      paragraph = [];
    };
    const closeList = () => {
      if (listType) output.push(`</${listType}>`);
      listType = "";
    };
    const closeFlow = () => { closeParagraph(); closeList(); };

    for (const line of lines) {
      const fence = line.match(/^\s*```\s*([\w.+-]*)\s*$/);
      if (fence) {
        if (codeFence) {
          output.push(`<pre><code${codeLanguage ? ` data-language="${escapeHtml(codeLanguage)}"` : ""}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
          codeFence = false; codeLanguage = ""; codeLines = [];
        } else {
          closeFlow(); codeFence = true; codeLanguage = fence[1] || "";
        }
        continue;
      }
      if (codeFence) { codeLines.push(line); continue; }
      if (!line.trim()) { closeFlow(); continue; }
      const heading = line.match(/^(#{1,4})\s+(.+)$/);
      if (heading) {
        closeFlow(); const level = heading[1].length;
        output.push(`<h${level}>${inline(heading[2])}</h${level}>`); continue;
      }
      const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
      const ordered = line.match(/^\s*(\d+)[.)]\s+(.+)$/);
      if (unordered || ordered) {
        closeParagraph(); const nextType = unordered ? "ul" : "ol";
        if (listType && listType !== nextType) closeList();
        if (!listType) {
          listType = nextType;
          const start = ordered ? ` start="${escapeHtml(ordered[1])}"` : "";
          output.push(`<${listType}${start}>`);
        }
        output.push(`<li>${inline(unordered ? unordered[1] : ordered[2])}</li>`); continue;
      }
      const quote = line.match(/^\s*>\s?(.*)$/);
      if (quote) { closeFlow(); output.push(`<blockquote>${inline(quote[1])}</blockquote>`); continue; }
      if (/^\s*(?:---+|___+|\*\*\*+)\s*$/.test(line)) { closeFlow(); output.push("<hr>"); continue; }
      closeList(); paragraph.push(line.trim());
    }
    if (codeFence) output.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
    closeFlow();
    return output.join("");
  }

  globalThis.KBotMarkdown = Object.freeze({ render });
})();
