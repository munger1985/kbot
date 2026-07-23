# 4.0 API、Schema 与数据生命周期

## API 契约与版本策略

Main API 的公开契约使用 `/api/v1`；Knowledge Core、Agent Runtime、AIOps、Model Serving 等服务间契约使用 `/internal/v1`。Model Serving 可额外发布受独立 API Key 保护的 `/api/v1/models`、`chat/completions` 和 `embeddings`，但模型管理始终属于内部契约。产品版本与接口版本独立，只有同一契约发生不兼容变化并需并行分流时才新增 `v2`。每个 API 定义请求/响应 DTO、分页、排序、错误码、幂等语义、超时行为和弃用策略；OpenAPI 是发布产物，并以消费者契约测试保护。

Core API 采用全新模型，不兼容旧 `/api/kb` 或任何 3.x Agent/Skill 契约。Main API 只发布 `/api/v1` 新路由，不提供旧接口 Adapter、双写、双读或协议转换；所有 Portal、APEX 页面、MCP 与外部调用方随 4.0 一起升级。

写操作使用 `Idempotency-Key` 或稳定来源键，返回资源状态与可轮询的 `job_id`。异步操作的状态、取消、重试、失败原因和最终结果必须可查询。错误响应不得泄露 SQL、文件路径、内部服务地址或凭据。

## 单一 APEX Schema 全量建设

所有对象继续位于 APEX 所需的单一 Schema；平台身份边界使用 `KBOT_PLATFORM_*`，领域表使用 `KBOT_KC_*`、`KBOT_AGENT_*`、`KBOT_OPS_*` 前缀。规范 DDL 位于 `database/oracle/<service>/`，按表 Owner 拆分。Main API 只读取平台拥有的 Domain Registry，不读取 KC/Ops 表。每个脚本记录所属服务、执行顺序、校验和、前置条件和 APEX 影响。

发布前建立 Schema Contract 清单：表、列、索引、视图、同义词、存储过程、触发器、APEX 页面/报表/LOV、后台脚本与 API 的依赖关系。禁止直接在生产库手工执行未登记 DDL。

4.0 初始发布只面向空的或已由环境管理员清理完成的 Schema，一次创建全部新对象。脚本不查询、转换或删除任何 3.x 对象；`KBOT_PLATFORM_DOMAIN`、模型配置、Collection 和其他业务配置全部重新创建。当前开发阶段的结构调整直接修改规范脚本并重建测试 Schema。Oracle Text/Vector 索引必须在入口启用前完成有效性检查。

## 4.0 初始数据

4.0 不把旧 KB/File/Chunk、Domain、模型配置或 Agent/Ops 状态映射到新模型。数据库初始化后为空，由 Portal/APEX 创建新的 Domain 与配置。KM Portal、普通上传和其他批准来源在 4.0 入口启用后按新请求创建 Bundle、Document、Version、Parse View、Evidence 和索引；这属于新的业务入库，不是 3.x 数据迁移。

验收数据使用专门准备的 4.0 Fixture 和 Golden Corpus，记录来源 ID、content hash、权限、job、parser/embedding/index version 与错误状态。验证比较的是 4.0 对新入库业务来源的完整性、权限过滤、Discovery/Document/Evidence Recall、页码定位和延迟，不比较或复刻旧 Chunk。

空库建库、Fixture 验收和一次性入口启用门禁见 [41_kbot4_step12_acceptance_release_and_cutover.md](41_kbot4_step12_acceptance_release_and_cutover.md)。

## 文件与对象存储生命周期

文件二进制不应以数据库 BLOB 或主机绝对路径作为长期契约。定义 `ObjectStorage` port，数据库只保存不可变 `storage_uri`、hash、大小、MIME、加密信息和保留策略。

上传先写隔离临时区，校验 hash、MIME、恶意内容和配额后原子发布；失败清理临时对象并记录审计。版本、解析产物、页图、OCR 原文和模型生成物分别有 URI、来源版本、保留期和访问权限。

删除采用 tombstone：先撤销检索可见性和下载权限，再异步清理索引、缓存和对象；到达保留期后物理删除。定期运行 orphan scanner 发现无数据库引用的对象和无对象引用的 Version，所有回收动作可审计、可重试。

## 版本、索引与可重现性

每个 Parse View、Evidence Unit、Discovery Object 记录输入 `content_hash`、parser/model/prompt/config snapshot、embedding model/version、索引版本和生成时间。新索引在候选版本构建完成、质量检查通过后才切换为 active；旧版本保留至回滚窗口结束。

需要重解析或更换 Embedding 模型时创建新 Job 和新 index version，而不是原地覆盖。支持全量/按 Collection/按 Document Version 重建，控制并发与成本，并在切换前运行离线评测和权限回归。
