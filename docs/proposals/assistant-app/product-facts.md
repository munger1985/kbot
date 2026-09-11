# 智能工作台的外部产品事实与验收前提

本文只记录本设计依赖的、已核对的外部事实与尚待 OCI 验证的假定。检索时间为 2026-09-10。

## 已核对：xAI 原厂 API

xAI 的 [X Search 官方文档](https://docs.x.ai/developers/tools/x-search) 明确说明：

- `grok-4.6` 可经 Responses API 使用 `{"type":"x_search"}`；
- X Search 能执行关键词、语义、用户和线程检索；
- 可传递 `allowed_x_handles`、`excluded_x_handles`、`from_date`、`to_date`、图片/视频理解选项；
- 结果可从 `response.citations` 读取引用；
- `allowed_x_handles` 与 `excluded_x_handles` 不能同时出现，账号列表各最多 20 个。

这证明 xAI 原厂协议有设计所需的 X Search 语义，但并不证明 OCI 已逐字段开放这些参数。

## 已核对：xAI 模型目录与文生图工具的边界

2026-09-10 实时读取 xAI 官方模型目录时，`grok-4.6` 的 `outputModalities` 为 `TEXT`，
而 `grok-imagine-image*` 被列为独立的图片生成模型。随后核对的
[图片生成工具文档](https://docs.x.ai/developers/tools/image-generation.md) 明确说明：
`grok-4.6` 可在 Responses API 中调用 `{"type":"image_generation"}`，由服务端的 Grok
Imagine 工具实际生成或编辑图片；响应返回 `image_generation_call`，其 `result` 是 Base64
图片字节。因此模型目录和产品入口应这样处理：

- 对“让 Grok 生成图片”的入口，可以绑定同一条 `grok-4.6` Responses 模型记录，并单独将
  `supports_image_generation` 验收为真；生成图片时，Grok 4.6 是编排模型，Grok Imagine 是
  上游内部工具，二者不能混称为同一个推理模型；
- 如果 OCI 另行开放直接的 Imagine 图片端点，则应创建独立模型记录；这属于另一种调用路径，
  不替代 Grok 4.6 的 `image_generation` 工具验收；
- X Search 与文生图的验收开关必须独立，因为一个工具被 OCI 开放并不意味着另一个也可用。

## OCI 假定与限制

Oracle 的 [预训练模型目录](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm) 当前公开页面列有 Grok 4.3、4.20 和 4.20 Multi-Agent，尚未列出 Grok 4.6。因此本设计不把 Oracle 文档的缺失误解为用户环境不存在模型，但也不把 xAI 原厂文档直接当作 OCI 契约。

本工作区安装的 OCI Python SDK `2.184.1` 的 Generic Chat 契约包含 `web_search_options`，
其选项仅包括检索上下文大小和用户位置；Generic Assistant Message 也包含 `url_citation`
注解。这说明 OCI Generic Chat 具备“Web 检索 + URL 引用”的候选路径，但 SDK 中没有 xAI
Responses 的 `x_search` Tool、账号白/黑名单或日期过滤字段，也没有 `image_generation` Tool。
因此不能把该字段等同为 X Search 或图片生成，更不能据此把模型目录的
`supports_x_search` 或 `supports_image_generation` 标为可用。

只有 Ashburn 的实际 API 验收全部通过后，模型目录才可登记以下能力：

| 能力 | 必须验证的结果 |
| --- | --- |
| `x_search` | `tools=[{"type":"x_search"}]` 被接受；可返回真实工具事件、至少一条可用引用及稳定的用量字段 |
| X 参数 | 账号白/黑名单互斥、日期范围、媒体理解参数按 OCI 返回的字段语义工作 |
| 文生图 | 明确的图像生成 Tool/端点可接受文本提示；返回可下载的图像字节或短时 URL，且能取得 MIME、尺寸与请求 ID |
| 流式响应 | 可区分工具进度、文本增量、引用/图片就绪、终态和失败；断线后能以 Run 状态恢复 |
| 错误和成本 | 不支持的 Tool、配额不足、内容拒绝、超时均有稳定错误码；响应包含可审计的用量或账单关联字段 |

验收失败时，应仅在模型能力矩阵中关闭对应入口，不能由前端做“看似可点、实际失败”的降级。
