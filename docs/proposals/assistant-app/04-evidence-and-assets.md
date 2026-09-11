# 04 · 证据、引用与图片资产设计

## 三类证据协议

引用是服务端可验证的结构化关系，不是模型输出的装饰文本。三个入口严格使用不同前缀，以避免用户把企业文档、查询快照和外部帖子混为一谈。

| 前缀 | 来源 | 结构化锚点 | 点击后的受控预览 |
| --- | --- | --- | --- |
| `[C1]` | KC Evidence | collection、bundle revision、document version、定位器 | 文档/页面/段落预览 |
| `[Q1]` | 问数 Query Result | query result、已发布语义模型版本、策略快照 | 查询条件、列、行预览、截断信息 |
| `[X1]` | Grok X Search citation | provider citation ID、外部 URL、作者/时间（若有）、检索时间 | 来源卡、受限摘要、打开外部 X |

生成回答前，服务端从 Artifact 建立允许引用集合，并只保留正文实际使用且可解析的标签。模型自行生成的未知标签须删除或使回答进入引用校验失败，而不能呈现为事实依据。

问文/问数的混合回答可同时有 `[C]` 和 `[Q]`；X Search Run 只能有 `[X]`；图片生成 Run 不产生上述事实引用。如此能维持既有 KM 引用语义而不污染 X 的外部线索。

## X Search 证据生命周期

X 的帖子、账号信息、图像/视频说明均是不可信外部输入。服务端将其置于明确的“不可信证据”容器中，并在系统提示中要求模型忽略来源内容中的指令、链接请求、角色重定义或数据外传要求。

`ExternalSourceReference` 至少保存：

```text
reference_id, research_run_id, provider, provider_citation_id,
canonical_url, author_handle?, published_at?, retrieved_at,
title?, excerpt?, content_hash?, display_order
```

默认只保存支持审计和重新展示所需的短摘要及哈希，不长期镜像完整帖子、媒体文件或账号档案。链接失效、删除帖、不可访问内容在预览中明确说明；不会用旧缓存冒充实时结果。保留期和最大摘要长度由 App 管理策略配置。

## 图片资产

图片生成结果是 App 私有的 Media Asset，二进制写入受访问控制的对象存储，数据库只保存元数据和对象键。建议核心字段：

```text
media_asset_id, generation_run_id, domain_id, owner_user_id,
object_key, sha256, mime_type, width, height, byte_size,
model_id, model_snapshot, prompt_revision_id, status,
created_at, expires_at?, source_run_id?
```

`model_snapshot` 和 `prompt_revision_id` 是可追溯信息，不保存 API Key、OCI 请求签名、上游短时 URL 或未经清洗的 provider 响应。下载走 Main API 的授权检查和短时签名 URL；共享链接必须另有明确的受众、过期时间和撤销状态。

图片详情卡统一显示“AI 生成”、模型显示名、生成时间、尺寸与提示词版本。它不保证版权、真实性、人物身份或可商用性；内容拒绝、策略限制和上游失败用稳定业务状态展示。

## 显式跨入口操作

首版不存在隐式跨入口工具调用。未来仅允许下列显式、可审计转交：

| 起点 | 用户操作 | 目标 | 必须保存 |
| --- | --- | --- | --- |
| X Search Run | “作为创作参考” | 新的图片草稿 | `source_run_id`、用户选中的来源 ID、生成的可编辑 Brief |
| 知识问答回答 | “作为创作参考” | 新的图片草稿 | `source_conversation_id`、`source_run_id`、用户确认后的摘要 |
| 图片资产 | “附加到知识资料草稿” | KC 审批流 | 新建待审批 Bundle，不自动发布 |

每种转交都须让用户编辑并确认目标输入，且新 Run 独立计费、独立授权、独立记录。
