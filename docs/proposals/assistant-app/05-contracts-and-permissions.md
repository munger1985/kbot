# 05 · API、内部服务、权限与验收设计

## 服务边界

浏览器只访问 Main API；Main API 校验 Portal API Key 和可信 App/Domain 上下文后，以服务凭据和短期、受众绑定的 AuthContext JWT 调用下游。Assistant App 不保存 OCI 密钥、不复用浏览器 API Key 下传、也不让前端指定模型端点。

```text
Browser
  → Main API /api/v1/apps/assistant/*
  → Assistant App /internal/v1/assistant/*
      ├─ Knowledge Retrieval / Agent Runtime / Knowledge Core / Data Query
      └─ Model Serving /internal/v1/responses/*
          └─ OCI Ashburn Grok 4.6
```

Assistant App 是资源编排、权限投影、运行记录和媒体资产的所有者。KC、Data Query、Agent Runtime 和 Model Serving 继续各自拥有事实、事务与持久化；Assistant App 不跨服务写 SQL。

## 公共 API 草案

所有路径均固定在 `/api/v1/apps/assistant`。请求的 Domain 来自业务 Token；除平台委托的 Domain 创建外，任何请求体中的 `domain_id` 都必须等于可信上下文，不能作为越权选择器。

| 资源 | 路径与操作 | 说明 |
| --- | --- | --- |
| 上下文 | `GET /access` | 当前权限、Domain、可用能力和安全显示信息 |
| Domains | `GET/POST /domains`，`GET/PATCH /domains/{id}` | 受控委托平台 Domain 生命周期 |
| KC | `GET/POST /knowledge-cores`，`GET/PATCH /knowledge-cores/{id}` | 代理/组合既有 KC 用例，不复制 KC 数据 |
| 问数模型 | `GET/POST /data-models`，`GET/PATCH /data-models/{id}`，`POST /data-models/{id}:publish` | 复用 Data Query 的语义模型/策略流程 |
| Agent | `GET/POST /agents`，`GET/PATCH /agents/{id}`，`POST /agents/{id}:enable` | 强制 KC Binding，启用前检查问数 Binding |
| 知识会话 | `POST /conversations`，`POST /conversations/{id}/turns`，`GET /runs/{id}`，`GET /runs/{id}/events` | 代理既有 Agent Runtime 的可恢复 Run |
| X Search | `POST /x-search/runs`，`GET /x-search/runs/{id}`，`GET /x-search/runs/{id}/events` | 创建独立研究 Run，不接受原始 OCI Tool JSON |
| 文生图 | `POST /image-generations/runs`，`GET /image-generations/runs/{id}` | 创建独立生成 Run |
| 图片资产 | `GET /media-assets`，`GET /media-assets/{id}`，`POST /media-assets/{id}:download-url` | 授权后短时访问对象存储 |
| 运行审计 | `GET /runs`，`GET /usage` | 按权限投影后的记录和用量，非供应商原始账单 |

每个创建型请求支持 `Idempotency-Key`；对可编辑配置使用 `If-Match: "rv-<row_version>"`。写入响应返回资源 ID、行版本和可恢复的 Run URL，异步执行返回 `202` 而非伪造同步成功。

## 内部契约

Model Serving 增加供应商中立的能力契约，而不是让 Assistant App 组装 OCI 专有 JSON：

```text
ResearchRequest
  model_ref, input, date_range?, included_handles?, excluded_handles?,
  enable_image_understanding?, enable_video_understanding?, trace_context
→ ResearchResult / ResearchEvent
  answer, citations[], tool_usage, model_usage, provider_request_id, status

ImageGenerationRequest
  model_ref, prompt, aspect_ratio, count, trace_context
→ ImageGenerationResult / ImageGenerationEvent
  artifacts[], model_usage, provider_request_id, status
```

模型目录的能力字段至少包括 `supports_x_search`、`supports_image_generation`、`supports_responses_streaming` 和 `capability_verified_at`。绑定页面只能选择 `ACTIVE` 且相应能力已验收的模型。能力缺失、配置缺失、提供商拒绝和配额耗尽须是不同的稳定业务错误。

## 权限矩阵

| 权限 | 可执行动作 | 不包含的权限 |
| --- | --- | --- |
| `assistant:access` | 进入 App、查看本人上下文 | 任何业务能力 |
| `assistant:knowledge_chat` | 使用获授权 Agent 聊天 | 创建 Agent、读取未授权 KC/问数结果 |
| `assistant:x_search` | 创建/读取自己或获授项目的 X Search Run | 修改模型绑定、读取其他 Domain 来源 |
| `assistant:image_generate` | 创建图片 Run、读取自己的资产 | 下载或管理他人私有资产 |
| `assistant:media_read` | 查看按 Domain 授权的图片资产 | 创建生成任务 |
| `assistant:domain_manage` | 创建/编辑/停用本 App Domain 的受控请求 | 跨 Tenant 或修改其他 App 的 Domain |
| `assistant:knowledge_core_manage` | 管理当前 Domain 的 KC 关联配置 | 绕过 KC 文档安全级别 |
| `assistant:data_model_manage` | 管理数据源、语义模型和 Policy 流程 | 访问明文凭据或任意 SQL |
| `assistant:agent_manage` | 创建、编辑、启用 Agent 及其绑定 | 赋予自己未拥有的权限 |
| `assistant:model_binding_manage` | 为 App 绑定已验收模型、配置额度 | 创建/读取供应商 Secret |
| `assistant:run_read` | 读取按授权投影的运行/用量 | 读取其他 Domain 的 Prompt、来源或图片 |

平台管理员仍掌管 App 生命周期、平台用户 Grant 和跨 App 边界；App 管理员只能在已获授 Domain 与权限范围内执行上述操作。

## 必须通过的验收

1. 未绑定 KC 的 Agent 无法启用，且不会出现在知识问答 Agent 列表。
2. 已绑定 KC、未绑定问数模型的 Agent 仅能走闲聊/问文；问数模型未发布或 Policy 失效时不能执行问数。
3. 三种引用在最终正文、`used_citation_labels` 与引用卡之间完全一致；未知 `[C]`、`[Q]`、`[X]` 标签被拒绝。
4. X Search 只有 OCI 能力验收通过后才可创建 Run；账号白/黑名单互斥、来源 URL 和外部线索标记可验证。
5. 文生图只在已验收模型上生成，产物存对象存储，下载和详情均再次校验 Domain/资产权限。
6. 浏览器请求不携带下游服务凭据或 OCI Secret；内部请求同时验证服务凭据和受众绑定 JWT。
7. SSE 中断后，知识会话、X Search 与文生图均能凭 Run 详情恢复终态，且不会重复扣费或重复发起提供商请求。
