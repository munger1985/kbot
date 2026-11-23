/* 两个建库抽屉共用检索能力预览；只说明配置，不声称已可检索。 */
(() => {
  "use strict";
  function textToImageLabel(item) {
    if (!Array.isArray(item.image_search_directions)) return "文搜图能力未登记";
    return item.image_search_directions.includes("TEXT_TO_IMAGE") ? "支持文搜图" : "不支持文搜图";
  }
  function imageSearchState(selected, model, direction) {
    const action = direction === "TEXT_TO_IMAGE" ? "用文字找相关图片" : "用一张图找相似图";
    if (!selected) return [false, "未配置", "未选择视觉 Embedding，不能" + action + "。"];
    if (!Array.isArray(model?.image_search_directions)) return [null, "能力未登记", "模型目录未提供有效的图片检索规格，请选择标明支持该能力的模型。"];
    if (!model.image_search_directions.includes(direction)) return [false, "模型不支持", "所选视觉 Embedding 不支持" + action + "。"];
    return [true, "已配置", "所选模型支持" + action + "，图片索引发布后可用。"];
  }
  function render(name, models, parseModels, catalog, escape) {
    const text = Boolean(models.embedding), visual = Boolean(models.visual_embedding);
    const model = catalog.find(item => Number(item.category) === 3 && item.model_id === models.visual_embedding);
    const visualParseNote = !parseModels.vlm
      ? "未选择 VLM，不能使用「视觉增强」解析。"
      : !parseModels.ocr ? "已选择 VLM，但「视觉增强」解析还需要 OCR。"
        : "已选择 OCR 和 VLM，「视觉增强」解析能否使用以审批时的模型校验为准。";
    const rows = [
      ["文搜文（关键词检索）", true, "基础能力", "不需要 Embedding。文本索引发布后，可以用关键词或错误码找内容。"],
      ["文搜文（语义检索）", text, text ? "已配置" : "未配置", text
        ? "文本向量索引发布后，可以找意思相近、用词不同的内容。" : "未选择文本 Embedding，不支持语义检索，仍可用关键词检索。"],
      ["文搜图", ...imageSearchState(visual, model, "TEXT_TO_IMAGE")],
      ["图搜图（找原图）", true, "基础能力", "不需要 Embedding，只能在已发布资料里找同一张原图。"],
      ["图搜图（找相似图）", ...imageSearchState(visual, model, "IMAGE_TO_IMAGE")],
    ];
    return `<section class="table-card">
      <header class="table-card-header"><div><h2 tabindex="-1" data-knowledge-capability-title>确认检索能力</h2>
      <p>知识库：${escape(name)}。以下按所选模型预览，实际检索须等待资料索引发布。</p></div></header>
      <table aria-label="知识库检索能力">
      <thead><tr><th scope="col">检索能力</th><th scope="col">配置状态</th><th scope="col">说明</th></tr></thead>
      <tbody>${rows.map(([label, configured, status, detail]) =>
        `<tr><th scope="row">${label}</th><td><span class="status-pill ${configured === null ? "status-warning" : configured ? "status-success" : "status-muted"}"><span aria-hidden="true">${configured === null ? "?" : configured ? "✓" : "×"}</span>&nbsp;${status}</span></td><td>${detail}</td></tr>`).join("")}</tbody>
      </table>
      <footer class="table-card-header"><div><p><strong>解析说明：</strong>${visualParseNote}</p></div></footer></section>`;
  }
  globalThis.KBotKnowledgeCapabilitiesReview = { render, textToImageLabel };
})();
