# 智能工作台 App 详细设计

## 设计结论

智能工作台（`assistant`）是与 KM、AIOps 同级的独立 App。它通过三个彼此可见、但默认不互相调用的入口提供能力：

1. **知识问答**：复用 Knowledge Core、Data Query 与 Agent Runtime，支持问文、问数、闲聊及其已有的受控混合路由。
2. **X 实时搜索**：由模型服务通过 OCI Grok Responses 能力执行；结果是外部实时线索，不是企业知识事实。
3. **文生图**：由模型服务通过 OCI Grok 图像生成能力执行；图片是有生命周期和访问控制的 App 资产。

本目录的五份设计产物是实现的单一设计基线：

| 产物 | 主题 | 实现决策 |
| --- | --- | --- |
| [01-shell-navigation.md](01-shell-navigation.md) | Shell、导航、主题、页面可见性 | 复用 KM/AIOps 的无框架静态 JS 页面形态 |
| [02-entry-pages.md](02-entry-pages.md) | 三个业务入口与页面状态 | 入口分离，不使用一个隐式混合的聊天框 |
| [03-resource-workflows.md](03-resource-workflows.md) | Domain、KC、问数模型、Agent | Agent 必须绑定 KC；问数模型只能绑定已发布版本 |
| [04-evidence-and-assets.md](04-evidence-and-assets.md) | `[C]`、`[Q]`、`[X]` 引用及图片资产 | 三类证据不混标；图片不写入数据库 BLOB |
| [05-contracts-and-permissions.md](05-contracts-and-permissions.md) | API、内部服务、权限、验收 | 浏览器只调 Main API；OCI 只由 Model Serving 调用 |

`product-facts.md` 是设计依据记录，不是第六项产品功能设计。

## 范围与非目标

首版范围是新 App 的页面、资源编排、权限、运行记录和模型服务适配契约。它不创建第二套 Knowledge Core、数据查询引擎、会话运行时或图片对象存储。

以下行为不在首版范围：

- 依据知识回答自动触发 X Search；
- 将 X 帖子自动写入 KC，或将其表述为企业已确认事实；
- 依据 X 帖子或知识问答内容自动生成图片；
- 让浏览器、Agent 指令或用户请求选择任意 OCI 端点、模型、Tool 或数据库；
- 用自由 SQL 替代已发布问数模型。

跨入口操作仅允许由用户显式发起。例如用户可在 X 搜索结果中选择“作为图片创作参考”，系统生成可编辑草案；用户确认后才发起图片生成。

## 上线前依赖

用户已确认 OCI Ashburn 可使用 Grok 4.6。本设计进一步假定其 Responses 协议与 xAI 的 `grok-4.6` 等价，且开放 `x_search` 与图片生成工具。这一假定必须通过 [product-facts.md](product-facts.md) 所列验收后才可在模型目录标记为可用；不能只根据模型名称启用入口。
