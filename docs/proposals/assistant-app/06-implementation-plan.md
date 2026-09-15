# 06 · X Search 与文生图实现计划

本文是 [README.md](README.md) 五份设计产物的实现切片，目标是让智能工作台通过 **Grok 4.6 Responses** 完成 **X Search** 与 **文生图**。它不修改 01–05 的产品决策，只把当前代码缺口补成可运行的内部服务、Main API、测试 UI 和单测。

真实 OCI Ashburn Canary（Phase G）不在本切片执行；能力开关保持为未验收，直到后续单独验收通过。

## 目标与非目标

本切片完成后必须具备：

1. Domain 角色绑定：只绑定 `ACTIVE` 且对应能力已验收的模型；用户请求不得携带 `model_id`。
2. X Search Run：独立研究任务，公开阶段可恢复，来源是外部线索 `[X]`，不是 KC 事实。
3. 文生图 Run：独立生成任务，图片写入 Assistant 本地对象存储，数据库只保存元数据。
4. 浏览器只调用 Main API；OCI 只由 Model Serving 调用。
5. 能力标志只能经 `record_capability_verification()` 写入，禁止模型 PATCH 改能力。

明确不做：

- 知识问答会话、Domain/KC/问数/Agent 向导；
- 隐式跨入口（依据知识或 X 结果自动生图、自动入库）；
- APEX / `integrations/apex/**`；
- 把 OCI Generic Chat 的 `web_search_options` 当作 X Search；
- 前端或 Agent 指令写死 `grok-4.6`；
- 本切片内对 Ashburn 发起真实 Canary。

## 调用链

```text
Browser UI (ui/assistant)
  → Main API /api/v1/apps/assistant/*
  → Assistant App /internal/v1/assistant/*
      ├─ 绑定、Run、事件、来源、提示词版本、媒体资产（本服务表）
      └─ Model Serving /internal/v1/responses/*
          └─ OciGrokResponsesAdapter → OCI Grok 4.6 Responses
```

Model Serving 不新建微服务，Responses 能力挂在现有 LLM 进程。

## 公开阶段与错误码

X Search：`ACCEPTED` → `SEARCHING` → `ORGANIZING_SOURCES` → `COMPOSING` → `COMPLETED` / `FAILED`

文生图：`ACCEPTED` → `GENERATING` → `COMPLETED` / `REJECTED` / `FAILED`

稳定错误码：

| 码 | 含义 |
| --- | --- |
| `MODEL_CAPABILITY_UNVERIFIED` | 模型未通过对应能力验收 |
| `MODEL_BINDING_MISSING` | 当前 Domain 未绑定该角色模型 |
| `PROVIDER_UNSUPPORTED_TOOL` | 上游拒绝 x_search / image_generation 工具 |
| `PROVIDER_QUOTA_EXHAUSTED` | 上游配额耗尽 |
| `CONTENT_REJECTED` | 内容安全策略拒绝 |
| `PROVIDER_TIMEOUT` | 上游超时 |
| `PROVIDER_UNAVAILABLE` | 上游不可用或配置不完整 |

## 数据归属

新增表均属 Assistant App，不写入 KC 对象存储，不把图片字节写入数据库 BLOB。

| 表 | 职责 |
| --- | --- |
| `KBOT_ASST_MODEL_BINDING` | Domain 角色绑定（`KNOWLEDGE` / `X_SEARCH` / `IMAGE_GENERATION`） |
| `KBOT_ASST_RUN` | X Search / 文生图 Run、租约、幂等键、`provider_request_id` |
| `KBOT_ASST_RUN_EVENT` | 公开阶段事件，追加写入 |
| `KBOT_ASST_X_SOURCE` | 经验证的 `[X]` 来源投影 |
| `KBOT_ASST_PROMPT_REVISION` | 文生图提示词版本 |
| `KBOT_ASST_MEDIA_ASSET` | 图片资产元数据与对象键 |

## 分阶段

### Phase A — 文档、契约、Schema

- 本文件，并在 README 中挂接。
- 扩展 `ResearchRequest.v1` / `ImageGenerationRequest.v1` 的公开事件与错误码。
- 新增上表及 Entity / Repository / UoW / `schema_manifest.json` / `SERVICE_TABLES`。

### Phase B — Model Serving Responses

LLM 进程新增：

- `POST /internal/v1/responses/research`
- `POST /internal/v1/responses/image-generations`
- `POST /internal/v1/models/{model_id}/capabilities:verify`（只提供代码，不自动执行）

`require_verified_capability()` 作为入口闸门。生产适配器为 `OciGrokResponsesAdapter`；`FakeGenerativeAdapter` 仅用于测试。`platform_clients` 增加 Responses 调用。

### Phase C — Assistant App 编排与 Worker

- `ModelBindingService`、`ResearchRunService`、`ImageGenerationService`
- Worker：`assistant_app.entrypoints.worker`，按 KM 的 poll/lease 认领 `ACCEPTED` 或租约过期的 Run
- 本地对象存储：`storage.local_object_storage_path`，语义对齐 KC 本地存储，但根目录独立
- 先落 Run 再调上游；若已有 `provider_request_id` 则不得二次调用
- 创建请求支持 `Idempotency-Key`

### Phase D — Main API BFF

扩展 `/api/v1/apps/assistant`：

- `GET /access` 附带绑定与能力摘要
- 模型绑定、X Search Run/事件、文生图 Run、媒体资产、Run 列表
- 用户体不接受 `model_id`；Domain 来自可信上下文

### Phase E — 测试 UI

接线 `x-search.html`、`image-generation.html`、`model-bindings.html`、`media-assets.html`、`usage-runs.html`。统一走 `runtime-config.js` + `assistant-api.js`。导航按权限显示。未验收/未绑定时不能创建 Run。

### Phase F — 与代码一起提交的测试

单元测试与 UI 静态契约。不打真实 OCI。

### Phase G — 后续（本切片不做）

Ashburn Canary 通过后，才允许调用 `record_capability_verification()` 把对应开关置为真。

## 实现约束

- HTTP → application → UoW/repository；Repository 不 `commit()`。
- 注释、文档、日志用中文；API 字段、错误码、协议值为英文。
- 不新增 3.x 兼容路径，不暴露 `/internal/v1` 到浏览器。
- 不重启服务、不 pull、不对 OCI 做真实验收，除非用户另有明确要求。

## 完成状态（2026-09-15）

Phase A–F 已落地。本切片不再重做契约、Schema、Entity/UoW、本地对象存储、LLM Responses 路由 / OCI 适配器、Assistant 编排、Main API BFF 与测试 UI。

| 阶段 | 状态 | 说明 |
| --- | --- | --- |
| A | 完成 | 本文件、公开阶段/错误码、`002_generative_runtime.sql`、Entity / Repository / UoW / `schema_manifest.json` |
| B | 完成 | LLM `POST /internal/v1/responses/research` 与 `image-generations`、`capabilities:verify`、`OciGrokResponsesAdapter`、`GenerativeResponsesClient` |
| C | 完成 | 绑定 / X Search / 文生图 / 媒体服务、内部路由、API lifespan、`assistant_app.entrypoints.worker`、本地对象存储 |
| D | 完成 | Main API `/api/v1/apps/assistant` 登录、改密、`GET /access` 附 bindings、X Search / 文生图 / 媒体 / Run / 模型目录 BFF |
| E | 完成 | `ui/assistant` 登录与权限导航；x-search / 文生图 / 模型绑定 / 图片资产 / 运行记录已接线；blob 预览不把 API URL 赋给 `img.src` |
| F | 完成 | 单元测试、UI 静态契约、OpenAPI 快照、拓扑 24 进程 / 6 worker、启停脚本含 Assistant Worker |
| G | 不做 | 真实 OCI Ashburn Canary；能力开关保持未验收 |

本切片验证（未重启、未 pull、未 commit/push、未打真实 OCI）：

- `tests/acceptance/check_openapi_contracts.py --write` 后无参校验通过：15 个快照，603 条路径。只保留本切片相关快照：`assistant_app_internal_v1.json`、`main_api_public_v1.json`、`model_llm_v1.json`。
- `python -m unittest tests.unit.assistant_app.test_generative_runtime tests.unit.main_api.test_assistant_app tests.contract.test_assistant_ui_static_pages tests.contract.test_process_topology`：21 passed。
- `node --check` 覆盖 `ui/assistant/js/*.js`。

### 接线顺序与约束

1. 浏览器只调 Main API；OCI 只由 Model Serving 调。
2. 用户体 `extra=forbid`，无 `model_id` / `domain_id`；内部 API 仍带 `domain_id`。
3. 创建支持 `Idempotency-Key`（缺省 `AuthContext.request_id`）；先查幂等，`IntegrityError` 后必须新开 UoW 再读。
4. 本地未绑定/未验收创建 Run → 422；Model Serving `require_verified_capability` → 409 `MODEL_CAPABILITY_UNVERIFIED`。
5. HTTP `create()` 返回 `(view, created)`：非终态 202，终态幂等命中 200。
6. Worker 认领后 commit 前 snapshot，再锁校验 `lease_token`；已有 `provider_request_id` 或 `upstream_attempted` 且非终态 → FAILED，不打上游。公开投影必须 pop `upstream_attempted`。
7. `GET /access` 调内部 `list_bindings`，不得用 `assistant:model_binding_manage` 卡住 access。
8. 图片 `object_key=media/{domain_id}/{run_id}/{asset_id}.{ext}`；`result_json` 只存 `asset_ids`；download-url 指向 Main API content。
9. 登录：`login_for_domain_name(..., domain_name="assistant_portal", app_id="assistant")`。UI `sessionStorage` 键 `kbot.assistant.session.v1`。导航按权限删除，不能只置灰。
10. 拓扑在 `assistant_app_api` 后增加 `assistant_app_worker`（kind=worker，无 port）；`[dependencies.assistant_app.llm] endpoint=model_llm timeout_seconds=300`。当前进程拓扑为 24 个进程、6 个 worker。
11. 本切片不重启、不 pull、不 commit/push、不对 Ashburn 做真实 Canary。

真实 OCI Ashburn Canary（Phase G）仍不在本切片执行。

## 导航韧性（2026-09-15 补丁）

登录后工作台正文可见、侧栏缺失，是因为 Shell 原先等 `GET /access` 成功后才注入 chrome，而 `/access` 又同步调用内部 `list_bindings`。Assistant App 未上线时，Main API 把 `AssistantAppClientError` 映射成 5xx，前端 toast 容器也还没挂上，用户只看到三张入口卡。

补丁约束：

1. UI 对齐 KM：先注入完整侧栏/顶栏，再用 `access.permissions` **删除**无权限项，不能只置灰。`/access` 失败时仍保留 chrome，并在已挂载的 toast 区报错。
2. BFF：`GET /access` 在 `list_bindings` 失败时仍返回 snapshot 权限；`bindings=[]`，capabilities 全 `ready:false`。导航不得被下游绑定查询绑死，也不得用 `assistant:model_binding_manage` 卡住 access。
3. 浏览器只调 Main API，禁止 `/internal/v1`；不写死 `grok-4.6`。
4. 本补丁不做 Phase G 真实验收；重启需用户明确要求。
