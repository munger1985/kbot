# 步骤 2 详细设计：运行、服务安全与可观测性

## 配置与启动

`kbot_app_knowledge.py` 是独立进程，但复用仓库统一配置加载、日志和 Oracle 连接池约定。KC 配置应使用独立命名空间，例如 `knowledge_core`，至少包含：监听地址/端口、Oracle 连接池、对象存储暂存/发布前缀、对象大小与 MIME 策略、暂存 TTL、Worker 租约/重试参数、服务认证材料、限流和审计开关。

`app_id` 与文本向量维度从 `base.toml` 固化读取；Embedding 模型由 Collection 绑定并从平台模型目录解析，不得由 Agent、Portal payload 或单次检索请求覆盖。Job payload 保存 Collection 模型身份快照以保证异步执行一致性。密钥、私钥、对象存储凭据和服务令牌只来自部署 Secret，不写入 TOML、日志、Receipt、Job payload 或 API 响应。

启动阶段依次校验：配置结构、Oracle 连通/DDL 版本、对象存储桶和最小读写权限、服务认证材料、必需索引/向量能力，以及配置模型与 Embedding 服务实际模型/维度、Oracle VECTOR 维度、ACTIVE Evidence/Discovery fingerprint 的一致性。任一硬依赖不满足则进程启动失败或 `/readyz` 为非 ready；不静默跳过不一致数据，也不以“降级到 V1 File/Chunk”启动。

## 健康检查与优雅停止

| 端点/行为 | 规则 |
| --- | --- |
| `GET /healthz` | 仅进程存活；不访问数据库，供 liveness 使用 |
| `GET /readyz` | 校验 Oracle、对象存储和关键配置；任一不可用即非 ready |
| 启动 migration 检查 | 只校验当前 KC Schema 版本，不在生产实例启动时自动执行 DDL |
| 优雅停止 | 停止接收新 multipart/claim；等待短请求完成；不主动取消其他 Worker 已持有租约 |

KC 不在 API 进程内执行长时间解析。Reaper、索引或清理 Worker 若与 API 同部署，也以独立运行角色和健康指标运行，避免 Web 实例重启影响任务恢复。

## 服务身份与最小权限

| 主体 | 最小权限 |
| --- | --- |
| APEX/用户经 Main API | Collection 管理、状态查询、以后受权限限制的 Discovery/Evidence |
| KM Portal | 仅配置允许的 Domain/Collection 的 `KM_ASSET` 受理与状态查询 |
| Parser Worker | 仅 claim 自己能力范围内的 PARSE Job、心跳和回传持有租约的结果 |
| KC 运维 Worker | 仅 Reaper、清理、索引/画像/关系等内部任务 |
| APEX 数据库直连 | 只读经 Collection Scope 过滤的 `KBOT_KC_V_*` 视图；无 KC 基表 DML |

服务间优先 mTLS；无法部署时使用轮换的短期服务令牌并限制 audience、服务名、Domain/Collection 范围和过期时间。内部接口即使在内网也必须认证，不能仅依赖路径前缀或源 IP。对象读取使用短期受控 URI 或存储 IAM 身份；URI 不应进入普通日志。

## 审计、日志与指标

每个外部请求生成/透传 `request_id`，并贯穿 Receipt、Job、Parser 回调、对象发布和日志。审计事件至少记录：actor、服务身份、Domain、Collection、来源键、Bundle/Revision、动作、结果码、路由版本和时间；正文、文件内容、下载令牌、完整栈和模型提示词不进入常规审计。

必须提供以下指标并按 Collection/来源类型/Worker 能力可聚合（禁止高基数 raw ID 标签）：

- multipart 接收量、拒绝率、暂存/发布耗时、孤儿对象与 Receipt 清理积压；
- Job 队列深度、claim 延迟、租约过期、重试次数、终态失败率；
- 按 MIME/View 的解析耗时、Evidence 数量、质量分布、回调批次失败；
- Revision `READY/PARTIAL/FAILED` 比例、从 `ACCEPTED` 到可检索的时延；
- Discovery/Evidence 查询量、延迟、空结果率和安全过滤拒绝率（步骤 5 启用）。

告警以用户影响为中心：Receipt 长时间 `RECEIVING/STAGED`、租约持续过期、Parser 某能力无 Worker、对象清理失败增长、Revision 失败率异常、当前可用 Revision 被全部隔离。日志统一结构化输出 `request_id/receipt_id/job_id/bundle_revision_id`，便于端到端追踪。

## 限流与数据保护

受理端按服务身份、Collection、总字节和并发 multipart 限流；Worker claim/回调按身份、能力和批次大小限流。输入文件使用允许 MIME、最大压缩比、最大解压大小、页数/Sheet 数、图片像素和 VLM/OCR 配额保护，避免压缩炸弹与成本失控。

保留策略区分事实与运行数据：Bundle/Version/Evidence 按 Collection Purge 删除；Receipt/Job/审计按合规期限归档后删除；暂存对象按短 TTL 清理。任何删除都先撤销检索可见性并保留必要审计，不以日志替代业务状态。
