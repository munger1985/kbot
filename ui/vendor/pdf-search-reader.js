// 两个平台共用：用发布证据的字符范围和版面坐标定位 PDF.js 文字层。
const assets = new URL("./pdfjs-6.5.438/", import.meta.url);
const MAX_PAGE_PIXELS = 12000000;
const MAX_PAGE_TEXT = 250000;
const MAX_CANDIDATES = 128;
let resources;

function loadResources() {
  if (!resources) {
    resources = Promise.all([
      import(new URL("pdf.min.js", assets).href),
      new Promise((resolve, reject) => {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = new URL("pdf-search-reader.css", import.meta.url).href;
        link.onload = resolve;
        link.onerror = () => { link.remove(); reject(new Error("PDF 阅读样式未加载")); };
        document.head.append(link);
      }),
    ]).then(([pdf]) => {
      pdf.GlobalWorkerOptions.workerSrc = new URL("pdf.worker.min.js", assets).href;
      return pdf;
    }).catch(error => { resources = undefined; throw error; });
  }
  return resources;
}

function normalize(character) {
  return character.normalize("NFKC").toLowerCase().replace(/[\s\u00ad\u200b\u200c\u200d\ufeff]+/gu, "");
}
const textKey = text => Array.from(text, normalize).join("");

function indexedText(layer) {
  const mapping = [], tokens = [];
  for (const div of layer.textDivs) {
    const node = div.firstChild;
    if (node?.nodeType !== Node.TEXT_NODE) continue;
    let offset = 0;
    for (const character of node.data) {
      const normalized = normalize(character);
      const entry = { node, start: offset, end: offset + character.length };
      tokens.push(normalized);
      for (let index = 0; index < normalized.length; index++) mapping.push(entry);
      if (mapping.length > MAX_PAGE_TEXT) return null;
      offset += character.length;
    }
  }
  return { text: tokens.join(""), mapping };
}

function candidateOffsets(text, needle) {
  const offsets = [];
  if (!needle) return offsets;
  let index = -1;
  while ((index = text.indexOf(needle, index + 1)) !== -1) {
    if (offsets.length === MAX_CANDIDATES) return [];
    offsets.push(index);
  }
  return offsets;
}

function rectanglesFor(mapping, start, end, paper) {
  const rectangles = [];
  const range = document.createRange();
  for (let index = start; index < end;) {
    const first = mapping[index];
    let last = first;
    while (++index < end && mapping[index].node === first.node) last = mapping[index];
    range.setStart(first.node, first.start);
    range.setEnd(last.node, last.end);
    for (const rect of range.getClientRects()) {
      if (rect.width < 0.1 || rect.height < 0.1) continue;
      rectangles.push([(rect.left - paper.left) / paper.width, (rect.top - paper.top) / paper.height,
        (rect.right - paper.left) / paper.width, (rect.bottom - paper.top) / paper.height]);
    }
  }
  return rectangles;
}

function within(box, rect) {
  // 允许字体上伸/下伸与解析框之间少量误差，但仍要求大部分字形位于源框内。
  const x = Math.max(0, Math.min(box[2] + 0.006, rect[2]) - Math.max(box[0] - 0.006, rect[0]));
  const y = Math.max(0, Math.min(box[3] + 0.009, rect[3]) - Math.max(box[1] - 0.009, rect[1]));
  return x * y >= 0.67 * (rect[2] - rect[0]) * (rect[3] - rect[1]);
}

function validBox(value) {
  return Array.isArray(value) && value.length === 4 && value.every(n => Number.isFinite(n) && n >= 0 && n <= 1)
    && value[0] < value[2] && value[1] < value[3];
}

function verifiedRanges(layer, paper, blocks, page) {
  const relevant = blocks.filter(block => block.anchor && block.locators.some(loc => loc.page_no === page));
  if (!relevant.length) return { matches: [], incomplete: false };
  const pageIndex = indexedText(layer);
  if (!pageIndex) return { matches: [], incomplete: true };
  const rect = paper.getBoundingClientRect();
  if (!rect.width || !rect.height) return { matches: [], incomplete: true };
  const { text, mapping } = pageIndex;
  const matches = [];
  let incomplete = false;
  for (const block of relevant) {
    const boxes = block.locators.filter(loc => loc.page_no === page && validBox(loc.bbox)).map(loc => loc.bbox);
    const parts = [];
    let whole = "";
    for (const segment of block.segments) {
      const needle = textKey(segment.text);
      if (needle && segment.matched) parts.push({ start: whole.length, text: needle });
      whole += needle;
    }
    if (!parts.length) continue;
    const findInBoxes = (needle, start, length) => {
      if (!boxes.length || !needle || !length) return [];
      return candidateOffsets(text, needle).flatMap(index => {
        const rectangles = rectanglesFor(mapping, index + start, index + start + length, rect);
        return rectangles.length && rectangles.every(r => boxes.some(box => within(box, r))) ? [rectangles] : [];
      });
    };
    // 先按完整段落确认偏移；完整段落在 PDF 中碎成多栏等情况，才尝试唯一的命中文字。
    const full = findInBoxes(whole, 0, whole.length);
    const alignedWhole = full.length === 1;
    const positioned = alignedWhole ? candidateOffsets(text, whole).find(index => {
      const rectangles = rectanglesFor(mapping, index, index + whole.length, rect);
      return rectangles.length && rectangles.every(r => boxes.some(box => within(box, r)));
    }) : undefined;
    for (const part of parts) {
      if (positioned !== undefined) {
        matches.push(...rectanglesFor(mapping, positioned + part.start, positioned + part.start + part.text.length, rect));
      } else {
        const candidates = findInBoxes(part.text, 0, part.text.length);
        if (candidates.length === 1) matches.push(...candidates[0]);
        else incomplete = true;
      }
    }
  }
  const unique = [...new Map(matches.map(r => [r.map(n => n.toFixed(5)).join(","), r])).values()];
  unique.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  return { matches: unique, incomplete };
}

function attachHighlight(paper, rectangles) {
  const overlay = document.createElement("div");
  overlay.className = "knowledge-pdf-highlights";
  overlay.setAttribute("aria-hidden", "true");
  for (const [left, top, right, bottom] of rectangles) {
    const mark = document.createElement("span");
    mark.className = "knowledge-pdf-hit";
    Object.assign(mark.style, { left: left * 100 + "%", top: top * 100 + "%",
      width: (right - left) * 100 + "%", height: (bottom - top) * 100 + "%" });
    overlay.append(mark);
  }
  paper.append(overlay);
  return overlay.firstElementChild;
}

export function mountPdfSearchReader(host, onPageCount = () => {}) {
  const status = document.createElement("p");
  status.className = "knowledge-pdf-status";
  status.setAttribute("role", "status");
  host.replaceChildren(status);
  let request, activeLoad, documentUrl, documentPromise, raster, layer, serial = 0, observedWidth = 0;
  let pageCount = 0;
  const cancelPage = () => { raster?.cancel(); layer?.cancel(); raster = layer = null; };
  const dropDocument = () => {
    const previous = activeLoad;
    activeLoad = documentUrl = documentPromise = null;
    if (previous) void previous.destroy().catch(() => {});
  };
  const signal = (state, message) => {
    host.dataset.pdfState = state;
    status.textContent = message;
  };

  async function render() {
    const config = request;
    if (!config || !host.isConnected || !host.clientWidth) return;
    const id = ++serial;
    cancelPage();
    host.replaceChildren(status);
    signal("loading", "正在渲染 PDF…");
    try {
      const pdf = await loadResources();
      if (id !== serial) return;
      if (documentUrl !== config.url) {
        dropDocument();
        documentUrl = config.url;
        activeLoad = pdf.getDocument({
          url: config.url,
          cMapUrl: new URL("cmaps/", assets).href,
          cMapPacked: true,
          standardFontDataUrl: new URL("standard_fonts/", assets).href,
          wasmUrl: new URL("wasm/", assets).href,
          iccUrl: new URL("iccs/", assets).href,
          enableXfa: false,
          isEvalSupported: false,
          maxImageSize: 36000000,
        });
        documentPromise = activeLoad.promise;
      }
      const documentRef = await documentPromise;
      if (id !== serial) return;
      pageCount = documentRef.numPages;
      onPageCount(pageCount);
      if (config.page < 1 || config.page > pageCount) {
        signal("out-of-range", "页码超出 PDF 范围，请重新输入。");
        return;
      }
      const page = await documentRef.getPage(config.page);
      if (id !== serial) return;
      const base = page.getViewport({ scale: 1 });
      const wantedScale = config.zoom ? config.zoom / 100 * (96 / 72) : Math.max(120, host.clientWidth - 32) / base.width;
      const viewport = page.getViewport({ scale: Math.min(wantedScale, 12000 / Math.max(base.width, base.height)) });
      const paper = document.createElement("div");
      paper.className = "knowledge-pdf-paper";
      paper.dataset.pdfPage = String(config.page);
      paper.style.width = viewport.width + "px";
      paper.style.height = viewport.height + "px";
      paper.style.setProperty("--total-scale-factor", String(viewport.scale * viewport.userUnit));
      paper.style.setProperty("--scale-round-x", "1px");
      paper.style.setProperty("--scale-round-y", "1px");
      const canvas = document.createElement("canvas");
      canvas.setAttribute("aria-label", "PDF 原文第 " + config.page + " 页");
      const outputScale = Math.min(devicePixelRatio || 1, Math.sqrt(MAX_PAGE_PIXELS / (viewport.width * viewport.height)));
      canvas.width = Math.max(1, Math.floor(viewport.width * outputScale));
      canvas.height = Math.max(1, Math.floor(viewport.height * outputScale));
      const text = document.createElement("div");
      text.className = "knowledge-pdf-text textLayer";
      paper.append(canvas, text);
      host.append(paper);
      await (raster = page.render({ canvas, viewport,
        transform: [canvas.width / viewport.width, 0, 0, canvas.height / viewport.height, 0, 0] })).promise;
      if (id !== serial) return;
      raster = null;
      let verified;
      try {
        layer = new pdf.TextLayer({ textContentSource: page.streamTextContent(), container: text, viewport });
        await layer.render();
        if (id !== serial) return;
        verified = verifiedRanges(layer, paper, config.blocks, config.page);
      } catch {
        if (id !== serial) return;
        verified = { matches: [], incomplete: true };
      }
      layer = null;
      const first = attachHighlight(paper, verified.matches);
      const isAnchorPage = config.page === config.hitPage || config.blocks.some(block =>
        block.anchor && block.locators.some(loc => loc.page_no === config.page));
      signal("ready", first ? (verified.incomplete ? "已定位到可核对的命中文字；其余仅定位到页。" : "已定位并标黄命中文字。")
        : isAnchorPage ? "当前页无法核实命中文字的位置，仅定位到页。" : "");
      host.scrollTop = 0;
      host.scrollLeft = 0;
      if (first) {
        const box = host.getBoundingClientRect(), hit = first.getBoundingClientRect();
        host.scrollTop += hit.top - box.top - Math.min(110, host.clientHeight / 4);
        if (hit.left < box.left || hit.right > box.right) host.scrollLeft += hit.left - box.left - 24;
      }
    } catch {
      if (id !== serial) return;
      cancelPage();
      dropDocument();
      host.replaceChildren(status);
      signal("error", "PDF 原文暂无法渲染。可以切到文字上下文核对。");
    }
  }

  const resize = new ResizeObserver(() => {
    const width = host.clientWidth;
    if (!width || width === observedWidth) return;
    observedWidth = width;
    if (request && (!request.zoom || host.dataset.pdfState === "loading")) void render();
  });
  resize.observe(host);
  return {
    get pageCount() { return pageCount; },
    update(config) {
      if (!config.url.startsWith("blob:")) throw new Error("PDF 预览只接受当前授权文件");
      if (request && config.url === request.url && config.page === request.page && config.zoom === request.zoom
          && config.hitPage === request.hitPage && config.blocks === request.blocks) return;
      if (request?.url !== config.url) { pageCount = 0; onPageCount(0); }
      request = config;
      observedWidth = host.clientWidth;
      void render();
    },
    destroy() {
      request = null;
      serial++;
      resize.disconnect();
      cancelPage();
      dropDocument();
      host.replaceChildren();
    },
  };
}
