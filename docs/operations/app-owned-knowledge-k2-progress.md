# App 独占知识能力：K2 实施与受控验收记录

日期：2026-11-05。状态：K2 基础组件已编码；知识检索 App 的真实 Oracle/PostgreSQL 定向验收均通过（各 18 项）。用户已授权 KC/知识检索 App 数据库操作，明确禁止修改 AIOps 数据库表；AIOps 真 Oracle 验收保留，整体 K2 G-B 不标记全量完成。两仓保持本记录一致。

依据：[实施计划](app-owned-knowledge-implementation-plan.md)、[接入详细设计](../proposals/app-owned-knowledge-integration-detailed-design.md)。本文不覆盖 K0/K1 历史记录，也不表示 K3–K7 或 R1 已完成。

## 1. 实现范围

两 App 分别新增 18 张冻结物理表的静态 Entity、逐表 Repository 和摄取聚合仓储，经各自真实 UoW 暴露。Oracle 使用 KBOT_KR_/KBOT_OPS_，PostgreSQL 使用 knowledge_retrieval_/aiops_；没有公共 KC 文档表落库、共享检索或动态 Tenant/Domain 分表。

- domain/knowledge_ingestion.py：可信 Scope、源版本、解析身份与 lease/fence 合同。
- adapters/knowledge_store.py：本 App 内容寻址源/解析/表示对象，流式大小和 SHA-256 校验、不可变提升、符号链接/路径拒绝；取消时等待在途文件 I/O 结束。
- application/knowledge_ingestion.py：登记、领取、提交回执恢复、续期、分页接收、事务提交后 ACK、ACK 回执记录及取消。外部 I/O 不持有业务事务。
- application/knowledge_projection.py：允许视图、精确原文映射及 App 策略注入；投影批次持久化、generation 准备/完整封闭及 outbox 处理。
- repositories/knowledge_ingestion.py：版本根行锁、重放/冲突、前向图像/引用依赖最终化、租约过期新 ParseRun、投影幂等、可靠 outbox 退避和 pin 保留/释放。
- composition/knowledge_ingestion.py：固定 App 组合根及双 JWT 内部源读取 API 工厂；不接受请求内 URL、表名、Python 路径或 Session。

规范数据保留原始事件及精确节点/图像/引用/关系身份。正文通过不可变事件对象定位，原图先复制到本 App 并验证摘要，再提交批次。结构和最终 Manifest 不完整、源版本/栅栏不一致时不创建可发布结果。forward reference 可暂存，必须在 STRUCTURE_FINALIZED 内闭合。

允许视图按 App 策略构造；新增/脱敏内容不能伪装成原文，必须保留等长精确来源映射。默认 reader 对未明确处理的同一图形多原图情况显式拒绝，场景可注入自己的 AllowedDocumentReader，不能默默选择一张图。

generation 的 PREPARING/READY 只表示投影准备进度，未切换 published_parse_run_id 或 active_generation_id。检索索引通道、真实查询、撤回/发布原子切换仍由 K3 实现；pin 目前实现保留事实，不启动任何业务对象 GC。

## 2. 计算协议与资源边界

knowledge_core 新增 domain/parse_jobs.py、adapters/parse_job_store.py、adapters/bounded_parse_executor.py、workers/compute_parser.py、application/parse_jobs.py、api/parse_jobs_router.py 和 composition/parse_jobs.py。

已在独立 ASGI 组合根实现并测试：

| 资源 | 行为 |
|---|---|
| POST /internal/v1/knowledge/parse-jobs | 严格解析冻结输入，202 回执，owner/主体/Domain/Tenant 从双 JWT 派生 |
| GET /{job_ref}/events | 有界分页，after_sequence/limit、next_cursor、status、事件摘要 |
| POST /{job_ref}/acknowledgements | 只确认已持久事件及匹配摘要，单调推进 |
| POST /{job_ref}:cancel | 封闭产出/确认并取消执行；App 先撤销本地 fence |
| GET /{job_ref}/artifacts/{artifact_ref} | 当前任务已声明的限时二进制流，句柄不转换为用户路径 |

源读取只通过固定 AppSource 注册表，使用本 App 的 /sources:read；客户端逐请求签发 audience/scope 限定的新 JWT，不跟随重定向，不接收回调 URL。源读取每块复核 App 当前任务身份和租约。

计算事件暂存仅用于任务交付，不提供公共知识查询。文件先 fsync、后推进元数据；支持该崩溃窗口恢复。跨 API 进程使用独占执行锁，避免把另一个进程的活跃任务当作重启中断。真正丢失执行者后明确 RUN_FAILED，由 App 建立新 ParseRun，不重跑旧非确定性代次。

执行器限制进程地址空间、CPU、输出文件、输入/事件/暂存预算和总墙钟，关闭时终止并等待子进程。ACK 窗口满后暂停读取 stdout，反压传递到子进程，而非无界收集全部输出。

本轮真实转换执行仅使用 K0 准入的 UTF-8 适配器。图像/引用链路使用明确的 synthetic 图文 fixture 验证持久化及交付协议；没有声称真实 PDF/Docling/OCR、多模态模型或生产规模质量已通过。复杂格式和真实模型准入/质量仍按 K5 执行。没有配置 reranker、rg 或新的模型类型。

新增组合根尚未替换现有线上业务入口、启动脚本或 API 快照；统一入口/worker 装配、暂存到期清理调度、旧 KC 路径退出及部署属于 K7。没有添加新旧双读/双写或兼容回退。

## 3. Schema 工件

KBot：

- database/oracle/knowledge_retrieval_app/003_knowledge_ingestion.sql；
- database/oracle/aiops_agent/014_ops_knowledge_ingestion.sql；
- 两份 schema_manifest.json 及 check_oracle_schema.py 的固定表所有权清单已同步；
- 新表只有 PK/FK/NOT NULL 和非唯一访问索引，没有 UNIQUE/CHECK；
- 离线 DDL 与 36 张 Entity 映射逐项核对；本轮只在获准的开发 Oracle Schema 新建知识检索 App 的 18 张规范表，未执行 AIOps DDL。

Ammolite：

- 36 张 canonical Entity 映射及各 App __all__ 已同步；
- scripts/deployment/init_postgresql.py 显式注册两个 App 的 entities.knowledge_ingestion；
- 未增加迁移、回填、旧字段兼容或运行时 Schema 探测。

本轮另外通过 scripts/deployment/knowledge_retrieval_k2.py 在本地 Ammolite PostgreSQL 开发库新建知识检索 App 的 18 张规范表。未运行全库初始化器，未修改已有业务表；这些空规范表保留用于后续开发，不随测试删除。KC 的旧数据库表、Prompt、模型配置及 AIOps 表均未修改。

两仓新增 tests/acceptance/check_app_owned_knowledge_schema.py，验证冻结列、36 个物理表、逐表仓储、FK 归属、约束政策及新增 API/Application 持久化边界。

## 4. 实际验证

解释器：/home/chris/miniconda3/envs/cube/bin/python。每仓显式加载本仓 packages/services 源码，没有将两份同名包混装到共享解释器。

| 检查 | KBot4 | Ammolite |
|---|---|---|
| K0/K1 算法单测回归 | 168 通过 | 168 通过 |
| test_parse_computation.py | 20 通过 | 20 通过 |
| test_app_owned_knowledge_storage.py | 9 通过 | 9 通过 |
| test_knowledge_computation_client.py | 5 通过 | 5 通过 |
| test_knowledge_retrieval_k2_scope.py | 5 通过 | 5 通过 |
| 固定 Schema/Repository 隔离离线检查 | 通过 | 通过 |
| 仓库架构边界 | 通过 | 通过 |
| 平台 Schema 离线检查 | Oracle 通过 | PostgreSQL 通过 |
| 真 PostgreSQL 摄取/恢复/竞态集成 | 不替代 Oracle | 18 通过，逐例覆盖两个 App |
| 真 PostgreSQL 知识检索定向开发库验收 | 不替代 Oracle | 18 通过；仅知识检索 App |
| 真 Oracle 知识检索定向开发库验收 | 18 通过；仅知识检索 App | 不适用 |
| AIOps 真 Oracle 验收 | 未执行；本轮禁止修改其表 | 不适用 |

计算测试实际运行受限 Python 子进程；包括启动中断、多个 API 执行者、输入超时、取消、事件落盘故障、背压、有期限制品、真实双 JWT 和合成 3000 段文档。大文档测试使用 128 MiB 地址空间硬上限，并断言实测子进程 RSS 小于 128 MiB；该结果不是 Docling 或生产 SLO。

前轮 PostgreSQL 双 App 验收使用仅绑定 loopback 的 postgres:16 一次性容器、内存数据目录以及每例随机隔离 Schema；没有连接现有业务库。18 项覆盖真实 Repository/UoW、提交前后 ACK、ACK 丢失、提交失败、计算回执丢失、新 fence、取消后源流拒绝、App/Domain/Tenant/主体隔离、并发消费、图像副本及前向引用、Manifest 失败不发布、允许视图、投影中断/幂等、outbox 退避、pin 保留与释放。fixture、模型故障均为合成，不使用真实文档或模型。

前轮验收后已停止并自动移除该测试容器及其内存合成数据，没有清理其他容器、卷或业务对象。本轮知识检索定向验收直接使用获准的开发库，边界见第 6 节。

主要复跑命令：

```bash
python -m unittest discover -s tests/unit/knowledge_algorithms -p 'test_*.py'
python tests/unit/knowledge_core/test_parse_computation.py
python tests/unit/knowledge_ingestion/test_app_owned_knowledge_storage.py
python tests/unit/platform_clients/test_knowledge_computation_client.py
python tests/unit/knowledge_ingestion/test_knowledge_retrieval_k2_scope.py
python tests/acceptance/check_app_owned_knowledge_schema.py
```

Ammolite 的 tests/integration/test_app_owned_knowledge_ingestion_postgresql.py 只接受显式 KNOWLEDGE_K2_TEST_DSN，且限定本机 knowledge_k2 测试库。

KBot 的旧双 App runner tests/integration/test_app_owned_knowledge_ingestion_oracle.py 仍只接受显式 KNOWLEDGE_K2_ORACLE_TEST_DSN、KNOWLEDGE_K2_ORACLE_TEST_USERNAME、KNOWLEDGE_K2_ORACLE_TEST_PASSWORD 和 KBOT_K2_TEST_ 前缀的空专用 Schema。它会建删 36 张表，本轮禁止执行，不能配置现有业务 Schema。PostgreSQL 旧双 App runner 同样不在本轮允许的执行范围内。未提供变量时的 skip 不计通过，不在日志中粘贴凭据。

额外全仓 Entity 所有权检查存在不在本项修改范围内的未满足项：platform_core 的 KBOT_PLATFORM_CASE_RESOURCE，以及 main_api 的 8 张 KBOT_NOTIFY_* 映射。未覆盖/修补无关在途修改，未把该全仓检查记为通过。本项两个 App 的新固定映射单独检查通过。

## 5. 退出判定和下一步

整体 K2 当前不能标记全量完成：知识检索 App 与 AIOps 的真库证据分别登记；只验证知识检索的 Oracle 适配器，不代表 AIOps 的 Oracle 事务、锁、恢复、竞态或跨 App 真库隔离已经验证。

本轮按用户的明确范围只执行知识检索 App 验收，不创建、写入或删除任何 AIOps 数据库表，也不为补齐门禁擅自新建数据库用户。既有文档、历史 Run、Prompt、模型配置与服务入口不改，未部署、重启、提交或推送。AIOps 的 Oracle 验收保留，不能用此前 PostgreSQL 结果或离线映射比较代替。

知识检索 App 的 K2 数据库受控门禁已通过：Oracle 26ai 与本地 PostgreSQL 18.3 各执行 18 项，包含真实 App UoW 及并发消费；本轮没有以 skip、mock 数据库或方言编译替代。每仓 K2 新增单测合计 39 项（原 34 项及本轮定向权限/结构核验 5 项），另有 K0/K1 回归 168 项通过。

K3 的真实检索/授权/跨文档引用、K4 的 Runtime 实际交付、K5 的真实模型/复杂转换质量、K6 的页面和 Domain、K7 的获准切换与清理仍独立保留，不提前声明完成。

## 6. 本轮知识检索专用开发库验收边界

- Oracle 通过用户指定的 kbotdev、密钥与 curve25519-sha256 连接，使用 /home/ubuntu/kbot4.0 的 typed 数据库配置；测试源码在独立临时目录执行，不覆盖线上代码。
- PostgreSQL 使用 /home/chris/ammolite_cube 的 typed 本地开发配置，不表示远端 Ammolite 部署已经更新。
- scripts/deployment/knowledge_retrieval_k2.py 默认只读；--apply 仅允许显式的 18 张知识检索表、索引及内部外键。已有表出现列/类型/非空/主键/外键/访问索引差异时拒绝自动修补；部分建表状态拒绝隐式补齐。
- 每条数据库写语句按编译目标表拦截；非授权表、其他 Schema、裸 SQL 写入、无范围 UPDATE/DELETE、DROP/TRUNCATE 均拒绝。没有运行全库 initializer。
- Oracle 类型核对读取 USER_TAB_COLUMNS，保留原生 JSON；时区索引只接受 Oracle 自动产生的精确 SYS_EXTRACT_UTC 等价表达式，不放宽任意函数索引。
- tests/integration/test_knowledge_retrieval_k2_development.py 复用原 18 项测试，仅装配知识检索 App。每例随机产生未占用的合成 Domain/主体（PG 同时隔离 Tenant），不向平台表登记测试 Domain；按逆外键顺序只删除该范围的合成行，核验清理完成，不 drop_all 或 DROP SCHEMA。
- 另一 App 仅比较离线实体表名与临时对象目录，没有读取/写入其业务数据。因此跨 App 真库隔离不计通过。
- 验收后再次只读核对：两库知识检索 18 张规范表均存在，表内剩余行总数均为 0；测试合成记录及临时对象已清理，规范表保留，没有删除既有业务数据。

在已核实并获准的目标环境中复跑：

```bash
python scripts/deployment/knowledge_retrieval_k2.py
# 只有明确批准新增这 18 张表后才使用 --apply；禁止运行全库初始化。
KNOWLEDGE_K2_DEVELOPMENT_ACCEPTANCE=knowledge_retrieval_only python tests/integration/test_knowledge_retrieval_k2_development.py
```

本轮额外执行的 Ammolite check_postgresql_entity_schema.py 是全库只读检查，未通过：AIOps 新知识表未创建（遵守本轮限制），另有 AIOps/通知/平台的在途 Schema 差异。未修改这些范围外对象，未把全库检查记为通过。知识检索 18 张表的定向结构核验通过。
