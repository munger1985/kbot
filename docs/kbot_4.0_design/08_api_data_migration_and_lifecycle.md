# 4.0 API、数据迁移与生命周期

## API 契约与版本策略

Main API 的公开契约使用 `/api/v1`；Knowledge Core、Agent Runtime、AIOps、Model Serving 等服务间契约使用 `/internal/v1`。Model Serving 可额外发布受独立 API Key 保护的 `/api/v1/models`、`chat/completions` 和 `embeddings`，但模型管理始终属于内部契约。产品版本与接口版本独立，只有同一契约发生不兼容变化并需并行分流时才新增 `v2`。每个 API 定义请求/响应 DTO、分页、排序、错误码、幂等语义、超时行为和弃用策略；OpenAPI 是发布产物，并以消费者契约测试保护。

Core API 采用全新模型，不兼容旧 `/api/kb` 或任何 3.x Agent/Skill 契约。Main API 只发布 `/api/v1` 新路由，不提供旧接口 Adapter、双写、双读或协议转换；所有 Portal、APEX 页面、MCP 与外部调用方随 4.0 一起升级。

写操作使用 `Idempotency-Key` 或稳定来源键，返回资源状态与可轮询的 `job_id`。异步操作的状态、取消、重试、失败原因和最终结果必须可查询。错误响应不得泄露 SQL、文件路径、内部服务地址或凭据。

## 单一 APEX Schema 变更治理

所有对象继续位于 APEX 所需的单一 Schema；平台身份边界使用 `KBOT_PLATFORM_*`，领域表使用 `KBOT_KC_*`、`KBOT_AGENT_*`、`KBOT_OPS_*` 前缀和独立 migration 目录。Main API 只拥有 API Key 配置、Domain Registry 等入口边界数据，不读取 KC/Ops 表。每个 migration 有唯一版本、checksum、作者、执行时间、前置条件、前向恢复说明和 APEX 影响标记。

发布前建立 Schema Contract 清单：表、列、索引、视图、同义词、存储过程、触发器、APEX 页面/报表/LOV、后台脚本与 API 的依赖关系。禁止直接在生产库手工执行未登记 DDL。

4.0 初始发布在隔离环境中创建完整新对象并一次启用，不与 3.x 对象进行兼容读写。旧 `KBOT_MD_DOMAIN` 仅作为一次性迁移来源，运行时认证只读取 `KBOT_PLATFORM_DOMAIN`；迁移保留数值 `APP_ID/DOMAIN_ID` 以满足 APEX。`expand → migrate → contract` 仅适用于 4.0 发布后的自身演进：先新增可选结构，再回填/校验/切换，最后在弃用窗口结束后删除旧的 4.0 对象。破坏性操作需备份、演练、维护窗口和明确回滚/前滚策略；Oracle Text/Vector 索引重建必须与业务写入和查询可见性协调。

## 知识资产重建

4.0 不把旧 KB/File/Chunk 表映射或同步到新模型。以原始来源（KM Portal/Metadb、受控文件存储和已批准外部系统）重新生成 Bundle、Document、Version、Parse View、Evidence 和索引；无法取得原始来源的内容只作为 3.x 归档，不进入 4.0 检索。

重建前建立来源清单和验收集，记录来源 ID、content hash、权限、重建 job、parser/embedding/index version 与错误状态。验证比较的是 4.0 对业务来源的完整性、权限过滤、Discovery/Document/Evidence Recall、页码定位和延迟，而不是追求旧 Chunk 的逐行复刻。

全量重建、Freeze Watermark、最终增量、来源对账和一次性入口切换的发布门禁见 [41_kbot4_step12_acceptance_release_and_cutover.md](41_kbot4_step12_acceptance_release_and_cutover.md)。

## 文件与对象存储生命周期

文件二进制不应以数据库 BLOB 或主机绝对路径作为长期契约。定义 `ObjectStorage` port，数据库只保存不可变 `storage_uri`、hash、大小、MIME、加密信息和保留策略。

上传先写隔离临时区，校验 hash、MIME、恶意内容和配额后原子发布；失败清理临时对象并记录审计。版本、解析产物、页图、OCR 原文和模型生成物分别有 URI、来源版本、保留期和访问权限。

删除采用 tombstone：先撤销检索可见性和下载权限，再异步清理索引、缓存和对象；到达保留期后物理删除。定期运行 orphan scanner 发现无数据库引用的对象和无对象引用的 Version，所有回收动作可审计、可重试。

## 版本、索引与可重现性

每个 Parse View、Evidence Unit、Discovery Object 记录输入 `content_hash`、parser/model/prompt/config snapshot、embedding model/version、索引版本和生成时间。新索引在候选版本构建完成、质量检查通过后才切换为 active；旧版本保留至回滚窗口结束。

需要重解析或更换 Embedding 模型时创建新 Job 和新 index version，而不是原地覆盖。支持全量/按 Collection/按 Document Version 重建，控制并发与成本，并在切换前运行离线评测和权限回归。
