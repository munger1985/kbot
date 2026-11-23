# App 独占知识能力：六项编码收尾与发布前置

日期：2026-11-06。适用 KBot4 与 Ammolite。本文记录源码与离线结果，不是上线或真实检索质量验收报告。

## 1. 六项交付

| 项目 | 正式落点 | 编码结果 |
|---|---|---|
| 知识检索 App | application/knowledge_library.py、knowledge_pipeline.py、knowledge_worker.py、composition/knowledge_business.py | 自有资料库／版本，流式幂等上传，后台领取／恢复，审核发布／撤回，Agent 绑定资料库 |
| Runtime | composition/knowledge_runtime.py、specialists/document/app_knowledge_skill.py、application/knowledge_delivery.py | 可信 Spec、Registry／DAG，精确版本授权，文本／图片实际字节交付，引用闭包与历史精确内容读取 |
| 合同与 API | platform_core/contracts、platform_clients/app_knowledge.py、knowledge_library.py、knowledge_computation.py；Main API 与 App 路由 | 公开 DTO／BFF、内部双 JWT／Run 委托与客户端同步，OpenAPI 为同一组正式路由 |
| 双前端 | KBot ui/knowledge_retrieval；Ammolite apps/portal-web 的知识 Workspace | 文搜文／文搜图／图搜图输入，主／关联结果、受控内容读取与续取；Domain／资料库切换清理状态，迟到响应隔离，上传同键重试 |
| AIOps | application/operations_documents.py、knowledge_search.py、knowledge_child.py、case_knowledge_projection.py | 只读取 AIOps 自有资料／案例；精确解析与索引代次，资格／脱敏／来源围栏，问答委托正式 Runtime 子执行 |
| 退出与发布工件 | 规范 Schema／manifest、生成 SQL、初始化、拓扑／安装、离线门禁与评测工具 | KC 为纯计算，不保留共享文档库、Bundle 或旧 Citation fallback；两个 App 分表且不跨 App 检索 |

正文“如图 1-1”与原图通过精确证据关系双向关联，图片命中可带引用正文。跨文档关系只接受可消歧 DOI／规范 URN 与显式引用标号，不根据标题或图号猜测身份。召回采用多通道融合、覆盖与预算选择，不引入 reranker 模型或 LLM rerank。

## 2. 可复现离线门禁

分别在两仓根目录执行，使用当前仓源码，避免同名 editable 安装串仓：

~~~bash
/home/chris/miniconda3/envs/cube/bin/python tests/acceptance/check_knowledge_foundation.py
~~~

脚本只运行合成／内存测试及架构、物理分表、规范 Schema 静态检查，不连接真实业务数据库或模型。包含受限 UTF-8 子进程、隔离 ASGI／SQLite fixture、故障注入和 AIOps 知识业务回归。新解析评测入口也用合成 UTF-8 进行了成功、缺失预期与空清单回归。

| 测试组 | KBot4 | Ammolite |
|---|---:|---:|
| 中立算法 | 168 | 168 |
| App 摄取 | 26 | 26 |
| 计算任务／执行器 | 26 | 26 |
| 客户端 | 5 | 5 |
| 基础接入／安全／配置／评测入口 | 90 | 90 |
| 知识检索 App／Runtime | 79，另 11 subtests | 48，另 4 subtests |
| AIOps 知识／案例／输出相关 | 672 | 669 |
| 既有 ColQwen 适配回归 | 不适用此测试文件 | 6 |

另有定向 Main API、配置／拓扑／页面契约、Oracle 生成 SQL 测试和跨项目 AIOps 语义检查；不计入上表。OpenAPI 只读复核：KBot 16 个快照／840 条路径，Ammolite 18 个快照／990 条路径。

离线浏览器已覆盖真实 Vue 组件与 KBot 页面，在四种宽度验证图文输入、关联结果、审核发布、上传失败同键重试、忽略 abort 的迟到响应隔离，以及 Domain 成功／失败恢复。对应命令：

~~~bash
node tests/contract/knowledge_page_alignment.mjs
node tests/contract/knowledge_retrieval_ui_style.mjs
~~~

脚本位于 KBot 仓；Vue 页面来自本地 Ammolite 源码，业务请求使用隔离 fixture。Ammolite Portal 的 npm run build 已通过类型检查与生产构建，仅有既有分块体积警告。这些结果不等于真依赖端到端测试。

## 3. 固定部署配置入口

单一部署 TOML 仅增加四个白名单段，由对应服务独占消费；不允许借此覆盖任意服务、数据库或路由：

~~~toml
[knowledge.computation]
cpu_seconds = 300
max_jobs = 2

[knowledge.knowledge_retrieval]
payload_tokens = 24000
payload_images = 8
payload_bytes = 16777216
representations = []

[knowledge.aiops]
representations = []

[knowledge.runtime]
payload_tokens = 24000
payload_images = 8
payload_bytes = 16777216
~~~

解析子进程默认使用 6 GiB 的虚拟地址空间硬上限（不是固定物理内存预留），为 Docling 模型权重映射保留空间；显式部署预算仍优先。子进程限制数值计算线程，资源耗尽单独报告，不归为协议错误。本次修复不需要改动系统级 TOML。实际部署仍需验证并发峰值与转换开销。App 和 Runtime 预算共同约束交付，模型输出 token 上限继续取自 Model Serving。

知识库 App 首次建索引和重建按批处理：每批最多 64 个检索单元，单批载荷逐单元核算（默认上限约 16 MiB）；同时保留整份文档的输入、单元数及单向量表示预算。建索引每批默认 180 秒，与解析截止时间独立；批次写入和最终校验也分别有时限。重试按当前文档版本、解析代次和索引规格定位同一代索引，校验已提交批次的数量和摘要后跳过重算，仍用租约防止过期 Worker 提交。未完成的索引不会被发布，不需要修改系统配置或表结构。

representations 需要完整 RepresentationSpec：model_ref、revision、processor_revision、input_template、dimension、kind、input_kind、similarity、normalization、directions。KBot 和 Ammolite 的推理请求均使用 model_id；模型生命周期与部署配置变更应共同审查，不能只检查 Agent 回答模型的引用。不得复制合成模型参数到真实配置。

本地文本 Embedding（Qwen／BGE）和本地 ColQwen2 视觉 Embedding 的模型版本、预处理器版本及输入模板标识均由 Model Serving 自动生成，无需手填。版本取自实际权重、配置和预处理制品；ColQwen2 的模型版本也覆盖基础权重及 adapter。文本输入模板取自实际生效的查询与文档格式及截断规则；视觉输入模板取自实际预处理规则及运行库版本。绑定和模型加载使用同一份规则：换权重、预处理器或输入规则后，旧索引不会被当成新向量空间使用。维度与视觉检索方向仍按模型规格登记；外部 API 模型继续使用目录中按供应商规格登记的版本和输入模板。

- 每个方向单独准入：TEXT_TO_TEXT、TEXT_TO_IMAGE、IMAGE_TO_IMAGE、IMAGE_TO_TEXT，方向必须指向该索引表示的模态。
- 模型回执须匹配冻结 revision／processor／template／维度／归一化／方向，空间不一致明确失败。
- 多向量保留原生 token／mask，不能均值池化后仍声称 MaxSim。
- 改变模型、处理器、模板或索引表示时建立新代次，重新构建、验证并经 App 发布；不能混用历史向量空间。

默认没有 OCR／语义表示模型，只有词法与图像 SHA 字节精确匹配。语义文搜图／图搜图的接口与算法链路已接通，实际可用性依赖对应模型配置及 K5 验证。界面能力以服务端声明为准；SHA 同图不等于语义相似或感知近重复。

## 4. 发布前必须满足

1. 冻结两仓代码、依赖、镜像、规范 Schema、模型／处理器与索引配置指纹，完成全部相关离线门禁。
2. 根据 topology 安装中立算法／摄取依赖，启动正式 App Worker；纯计算进程仅访问固定 App 源端口及有期限暂存制品，不访问业务知识表。
3. Oracle 规范 DDL／manifest／生成 SQL，或 PostgreSQL canonical ORM／初始化必须一致。生成重建 SQL 不代表允许执行，禁止对全库或无关 App 初始化。
4. 按 App 独立批准数据库／对象／索引变更范围，验证备份与一致回退能力。没有该批准就不得处理历史 KC 数据或切换流量。
5. 执行 K5 真实文档／模型质量与规模实验，以及 K6 真依赖上传、解析、审核发布、三类检索、图文关系、问答交付、撤回与 Domain 隔离。
6. 通过后再执行 K7 发布，检查实际进程／端口／近期日志并保存记录。回退需要恢复代码、数据库、对象与索引的一致快照，不能只回退代码。

当前 ExactIndex 是有界全授权精确基线，不是大规模 ANN。超限显式失败，不能截断后声称已检索全部；支持规模和“最佳效果”只能由真实评测确认。可选 ripgrep 默认关闭，启用需独立安装、受控绝对路径与授权文本投影，并增加专项质量／成本门禁。

## 5. 真实状态与已知边界

六项离线编码阶段未进行真实数据库、模型、业务文档测试；未部署、重启、注册模型、修改 Prompt 数据库、切换线上流量、清理旧数据或修改 AIOps 数据库。历史 K2 定向实库记录不算该阶段重跑。后续授权的 Schema 操作见第 6 节。

旧公共 KC／Citation 实现从活跃源码删除，以 Git 历史恢复代码；本轮没有删除数据库对象或历史文件制品。新版本不提供兼容路由、双读双写、静默 fallback 或自动回填，旧历史 Run 的数据保留／访问策略必须在发布时单独批准。

上表只证明相关离线范围通过，不能宣称全仓回归全绿。扩大检查发现的非本任务失败包括既有通知异步测试依赖／实体登记、AIOps 计算图／显示字典／诊断断言、KBot 既有运维脚本清单／通知升级目录断言和其他 App 的页面样式断言；未以跳过或修改这些模块伪造全仓通过。

设计、详细设计、实施计划和本说明按此前要求保留本地；后续授权的提交／推送见第 6 节。外部验收与发布仍按实施计划 K5／K6／K7 逐项留证。

## 6. 后续授权：kbotdev Schema 与代码推送

2026-11-06，用户明确要求登录 kbotdev 升级 Schema 后推送代码。连接使用指定用户、密钥及 curve25519-sha256；没有拉取线上仓库或重启进程。

- 仅对 KBot Oracle 的知识检索 App 执行 3 条 DDL：创建 KBOT_KR_KNOWLEDGE_LIBRARY、IX_KR_LIBRARY_DOMAIN 索引，向 KBOT_KR_AGENT_VERSION 增加可空 LIBRARY_ID RAW(16)。
- 现有 18 张基础表经最新 K2 只读工具核验，missing_tables=[]；新资料库列和索引经升级器再次核验。
- 32 个 Agent 版本的旧字段值全部保持原样，其中 5 条非空 KC 绑定没有自动映射。新 LIBRARY_ID 不回填；新版本启用后须显式创建／绑定资料库。
- 旧 KNOWLEDGE_CORE_ID／DO_RERANK 列和 KC 历史表仅保留旧数据，不在新源码读取或双写；清理不在本次操作内。DO_RERANK 既有默认值已经是 0，不需要改动。
- AIOps 与 KC 的元数据指纹前后完全一致，没有执行 DML、删表、删除旧列、资料搬迁、Prompt／模型更新或 AIOps Schema 变更。
- 远端私有审计目录：/home/ubuntu/db-schema-upgrades/knowledge-app-cjcnNmfr，包含 before.json／plan.json／after.json／completed.json，文件权限 0600。Oracle DDL 自动提交；升级器记录实际完成语句，不伪称事务回滚。
- 新增 scripts/deployment/upgrade_knowledge_retrieval.py：默认只读计划，写入必须提供匹配计划指纹和私有审计目录；仅允许固定知识检索对象的非破坏式扩展。6 个纯离线测试通过。

- Ammolite PostgreSQL 根据当前 canonical ORM 定向创建 19 张 knowledge_retrieval_knowledge_* 表（18 张基础表加资料库表），并向 knowledge_retrieval_agent_version 增加可空 library_id UUID，共 89 条建表／索引／外键及补列 DDL，在单一事务中完成；原 Agent 版本为 0 行。
- PostgreSQL 升级后重新运行只读计划，missing_tables=[]、ddl_count=0；19 表列名／类型／可空性与 ORM 完全匹配，旧绑定及 KC／AIOps 列和索引元数据指纹保持不变。
- PostgreSQL 一次性部署工件由当前 ORM 限定表清单生成，仅保存在上述远端私有目录，不向代码仓库增加迁移／兼容路径。审计为 postgres-before.json／postgres-plan.json／postgres-after.json／postgres-completed.json；postgres-completed.json 确认 committed=true。前述 scripts/deployment/upgrade_knowledge_retrieval.py 和 before.json 等为 KBot Oracle 的独立工件。

代码提交包含生成 OpenAPI 合同快照，不包含 docs 下的设计／实施文档。Schema 完成不等于新应用已部署；线上仍需按批准流程更新代码及完成真实场景验收。

代码已提交并推送，且通过 git ls-remote 核验远端分支与本地 HEAD 一致：KBot kbot4.1.1 = a607b2eaa3f7db13945606ed29d36ae6ad4511c9；Ammolite dev = d96c9c4a1476bec51f8995a90d53d48ebb73208c。推送后两仓工作树仅余 docs 下未提交的设计／实施说明，没有暂存残留；未在 kbotdev 执行 git pull、部署或重启。
