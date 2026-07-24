# 视觉 Embedding 与聊天多图检索

## 目标与边界

4.0 恢复 3.x 的以图搜图能力，但不恢复 `extracted_images`、KB 或 Chunk 依赖。视觉索引属于 Knowledge Core；查询图片属于 Conversation 的临时输入；Visual 模型由 Model Serving 托管。VLM 用于理解图片和生成描述，Visual Embedding 用于相似度检索，两者不可互换。

Collection 可绑定一个不可变的 `visual_embedding_model_id`。未绑定时，该 Collection 仍可正常进行文本解析和检索，但不会创建或接受视觉向量。不同 Collection 可使用不同视觉模型，查询时必须按 Collection 分组编码，禁止把不同模型产生的向量放进同一排序空间。

## 入库与索引

Parser 保留 Docling 生成的整页截图和 Figure 原图，并提交为 `KBOT_KC_VISUAL_ASSET`：

- `PAGE`：保留纯视觉解析在复杂版式、扫描件和图表页面上的优势。
- `FIGURE`：支持图、照片和局部图表的精细命中。
- `EVIDENCE_ID` 可关联 Figure 的正文 Evidence；页面资产通过 Document、Version、页码回到可引用正文。

Parser 只写不可变图片、定位、描述和哈希，不生成检索向量。INDEX 阶段读取 Collection 绑定模型，写入模型 ID、服务名、配置指纹和向量。Parse View 替换时，旧视觉资产随旧视图退出检索。

## 聊天查询

公开接口 `POST /api/v1/conversations/{conversation_id}/turns/multipart` 接受 1–8 张 PNG、JPEG 或 WebP；单图不超过 16 MiB，总计不超过 32 MiB。Agent Runtime 将原图写入内容寻址的附件存储，Conversation Item 仅保存 URI、哈希、大小和 MIME，历史渲染不保存 Base64。

Document Skill 对每张图片分别召回，再使用 RRF 按 Bundle Revision、Document 和页码融合去重。视觉候选优先进入第二阶段 Evidence 检索，并与自然语言 Discovery 候选合并。最终回答仍必须引用 Evidence；视觉相似度只负责选候选，不能单独成为事实依据。

聊天图片采用两条相互独立的可选路径：

| Query VLM | Collection Visual Embedding | 行为 |
|---|---|---|
| 已配置 | 已配置 | 同时执行图片转文字检索和图搜图，再融合候选 |
| 已配置 | 未配置 | 仅执行 VLM 图片转文字后的文本检索 |
| 未配置 | 已配置 | 仅执行图搜图 |
| 未配置 | 未配置 | 忽略图片，继续处理用户文字，并返回前端提示 |

`Citation Pack.query_plan.image_processing` 分别记录
`vlm_text_search` 和 `visual_search` 状态，取值包括 `EXECUTED`、
`PARTIAL`、`SKIPPED_NOT_CONFIGURED`、`FAILED`。降级原因进入
`warnings`，并投影到 `retrieval.completed` SSE Trace 和最终回答历史。
配置存在但调用失败时只降级该图片路径，不应中断仍可用的文本路径。

## 解析可选路径

解析侧同样独立判断两类模型：

- 配置 Parser VLM 且解析策略允许时，执行 Figure 描述和自适应整页视觉解析；
  未配置时保留 Docling/OCR 结果，不调用 VLM。
- 配置 Collection Visual Embedding 时，Parser 导出 PAGE/FIGURE 原图，
  INDEX 随后生成视觉向量；未配置时不导出视觉索引资产。
- 两者同时配置时两条路径都执行；任何一条缺失都不会阻断另一条或文本解析。

解析质量报告的 `image_processing` 字段记录 `vlm.enabled`、
`vlm.skip_reason` 和 `visual_embedding.status`，用于任务详情和运维验收。

生产部署时，Agent Runtime API 与 Worker 必须共享附件对象存储；本地开发使用共享目录。KC 返回的内部 `payload_uri` 不应直接暴露给浏览器，后续由 Main API 提供受控下载或短期签名 URL。
