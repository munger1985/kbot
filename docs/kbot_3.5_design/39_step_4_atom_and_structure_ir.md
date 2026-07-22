# 步骤 4 详细设计：Atom IR 与 Structure IR

## Atom IR：忠实统一底层提取结果

Atom 是最小可定位提取单元，不等于检索切片。`AtomNormalizer` 将 Docling、OCR 或受控视觉结果统一成版本化 JSON；它可以规范空白和坐标，但不能改变事实内容或推断章节关系。

```json
{
  "ir_version": "kc-atom/v1",
  "document_version_id": 301,
  "generator": {"name": "docling-adapter", "version": "..."},
  "pages": [{"page_no": 1, "width": 595, "height": 842}],
  "atoms": [{
    "atom_id": "atom:...",
    "source_ref": "#/texts/17",
    "atom_type": "TEXT",
    "content_text": "可从源文件核验的文本",
    "locators": [{
      "page_no": 1,
      "bbox": [0.10, 0.20, 0.85, 0.26],
      "original_bbox": {"l": 59.5, "t": 168.4, "r": 505.7, "b": 218.9, "coord_origin": "TOPLEFT"}
    }],
    "reading_order_hint": 12,
    "original_label": "text",
    "style": {"font_size": 11, "bold": false},
    "confidence": 0.97,
    "provenance": [{"extractor": "DOCLING", "source_ref": "#/texts/17"}]
  }]
}
```

`atom_type` 至少支持 `TITLE_CANDIDATE/TEXT/LIST_ITEM/TABLE/PICTURE/CAPTION/FORMULA/CODE/FOOTNOTE/HEADER/FOOTER`。`atom_id` 由 Document Version、规范化 `source_ref` 和类型确定性生成；缺少源引用时使用页面、规范 bbox、局部序号与内容 hash。重试同一 View 必须稳定。

每个 Atom 使用 `locators[]` 保留全部 Docling provenance，兼容跨页来源；分页格式同时保存原始坐标和统一后的 `page_normalized_top_left` 坐标。DOCX/Markdown 等非分页格式没有页面几何时使用 `logical_ref=Docling self_ref`，不伪造页码或 bbox。页眉页脚不能直接删除，应标注 `repeated_region_key` 和排除理由，以便审计或修改策略后回放。

## Reading Order：先处理版面，再判断标题

`ReadingOrderResolver` 以页面区域、栏簇、Docling 顺序提示和几何关系生成全文稳定序列。表格、图片与其 caption 作为邻接约束；页眉页脚从正文序列隔离；跨页时利用页面边界、句法连续性和样式连续性建立 `continuation` 候选。任何低置信度排序都写入质量报告，不能静默采用。

## Structure IR：重建而非修改旧树

`OutlineResolver` 必须从有序 Atom 重新构建结构树，不能只修改标题 level。`SemanticBlockBuilder` 再将同一语义单元的 Atom 组合成段落、列表、表格或图文块。

```json
{
  "ir_version": "kc-structure/v1",
  "nodes": [{
    "node_id": "section:...",
    "node_type": "SECTION",
    "parent_node_id": "document:...",
    "ordinal": 4,
    "atom_ids": ["atom:heading-4", "atom:paragraph-8"],
    "heading": {
      "atom_id": "atom:heading-4",
      "text": "2.1 部署条件",
      "level": 2,
      "confidence": 0.94,
      "reasons": ["numbering:2.1", "style_cluster:h2", "toc_match"]
    },
    "heading_path": ["2 部署", "2.1 部署条件"],
    "page_range": [3, 4],
    "continuation_of": null,
    "decision_provenance": {"resolver_version": "outline/v1"}
  }]
}
```

节点类型至少包括 `DOCUMENT/SECTION/PARAGRAPH/LIST/TABLE/FIGURE/CAPTION/FORMULA/CODE_BLOCK/FOOTNOTE`。章节节点可以嵌套任意深度；语义块保留有序 `atom_ids` 和精确 locator 集合，而不是只保存起始页。

## 标题与章节推断

标题候选综合全文特征，不依赖单页长度阈值：编号语法、字体/粗细样式簇、上下留白、目录匹配、同级重复模式、句法特征和相邻块类型。解析器以动态规划或受约束 beam search 选择全局层级序列，对非法跳级、孤立标题、样式突变和编号倒退施加惩罚，并输出每个决策的置信度。

低置信度区域可请求 VLM 在给定 Atom 集中选择标题、层级或阅读顺序。VLM 不得生成新标题文字，也不得返回不存在的 Atom；结果仍要经过树不变量校验。由图像内容产生的文字描述必须作为独立派生节点，不进入原文标题树。

## 必须满足的不变量

- 除结构引用外，每个正文 Atom 恰好归属一个主要语义块，不丢失、不重复消费。
- 树无环、父节点先于子节点，全文 ordinal 稳定；章节层级跳变必须有显式修复记录。
- `content_text` 拼接后与参与的 Atom 文本一致；结构解析不得新增来源事实。
- 每个 Evidence 均可追溯到一个或多个 Atom，且所有 Atom locator 可映射回原文件。
- 表格标题、表头、caption 和正文分别保留 Atom 身份，组合时不丢失行列定位。
- 同一输入、策略和组件版本必须生成相同 IR hash；非确定模型输出需记录模型、prompt、响应 hash 与随机性参数。

这些不变量在生成 Evidence 前执行。违反内容覆盖、树合法性或定位完整性属于硬失败；低标题置信度、疑似阅读序异常属于可触发局部增强的软失败。

## 当前实现基线

IR 已落在 `knowledge_core/parsing/ir.py`，采用不可变 dataclass，并提供规范 JSON 指纹。当前硬校验包括定位合法性、Atom/来源引用唯一、提取 provenance、唯一 Document 根、无环父子树、标题来源与层级、Atom 单一归属和正文 Atom 全覆盖。`docling_adapter.py` 负责 label 映射、表格原文、全部 provenance、坐标统一、Sheet/cell offset 和页眉页脚标记；OCR/VLM 描述生成独立 `VISUAL_DESCRIPTION` Atom，不覆盖来源内容。`contracts.py` 统一 Evidence key、联合指纹、工件清单和质量报告校验。
