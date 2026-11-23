# 本地 PG / Oracle Schema 对齐验收（2026-11-17）

## 2026-11-18：无损补齐本地开发库

仅操作应用本地开发配置指向的 PostgreSQL `localhost:5432 / ammolite / ammolite` 和 Oracle `127.0.0.1:1521 / KBOT4 / KBOTDEV`；未操作 Docker 测试库。按当前规范各增补数据保护计划、运行两张表及相应约束和索引，不初始化或重建已有表。

两库各三条既有人工恢复档案，按主键、资源、域及行版本与下方已校验的原始备份逐条匹配，PG 另核对租户。补回四个备份策略字段的原始值并设为规范非空；已有档案的其他字段前后逐行一致。没有删除数据、改写其他配置或使用默认值替代历史值。

Oracle 首次建运行表遇到 `ORA-02329`。规范 DDL、ORM 和生成工件均已修正：保留带时区的时间列，去掉不受支持的业务唯一约束；KBot 在现有认领短事务内对运行表加锁后检查同火次和占用键，继续保护状态未知的任务。未跨备份/恢复工具调用持有此锁。修正后补建通过；Oracle Schema 静态检查、KBot 架构边界及 98 项定向测试（另 5 个子测试）通过。

复验：两库增补预演都无待执行项，PG 默认值无差异。Oracle 全服务 Entity 检查通过（196 表、2768 列）；PG Entity 检查仅报告额外表，没有规范表、列、类型或默认值差异。严格 Catalog 检查仍报告两库各六张旧试行空表（`backup_fire`、`backup_plan`、`backup_schedule`、`protection_location`、`protection_object`、`protection_reservation`，各自带平台前缀），有意保留，未删表。未运行真实备份或恢复演练、重启、编译或发布。

## 代码回退后的适用范围

以下是 2026-11-17 当时的实库操作记录，不再是当前源码的验收结论。数据保护试行实现现已从两仓源码回退；本轮没有连接或修改数据库。按下文记录，本地 PG、Oracle 此前各加入 6 张试行表，并删除各自旧人工恢复配置需要的 4 个备份策略字段。回退后源码恢复了这些字段，也从清理清单排除了它们；本地库需要用户另行确认恢复方案，不能直接用初始化、默认值回填或清空表代替找回已有配置。

## 最新结果：已获准清理，未新增备份

用户回复“不用备份”后，按前一轮列明的固定范围完成清理。未创建新备份，
没有删除前阶段已存在的备份，也未运行全库初始化、CASCADE、服务重启或模型种子。

- PG 删除 15 张退役表及其中 184 条旧记录（含 32 份文档登记），删除 12 个旧字段、
  5 个额外普通索引；清除 2 条旧 bundle 索引记录后，补齐 3 个 UUID NOT NULL 必填字段。
- Oracle 仅删除报告列明的 8 个旧字段，没有删除任何模型、凭据或配置行。
- PG 的 20 张、Oracle 的 12 张受保护表按规范列投影逐表比较内容指纹，一致。
  PG 38 个模型、Oracle 30 个模型、API Key、账号密码凭据、授权、
  知识库、原始资产及资产版本均保留；未删除任何物理资料文件。
- PG 全库 Entity/字段/默认值验收通过：259 表、3781 列；Catalog 验收通过：259 对象。
- Oracle 全库 Entity/字段验收通过：200 表、2824 列；Catalog 验收通过：214 对象。
- 清理后重新核对规范主键、外键、索引，PG/Oracle 均无缺失、额外或实际定义偏差；
  Oracle 的 LOB 内部索引及等价表达式改写不属于额外业务索引。
- 重复预演两个清理入口均无待执行语句。51 个定向测试通过
  （PG unittest 19、PG 知识配置部署 pytest 3、Oracle unittest 29），两仓架构检查通过。
- 原始模型引用接口返回 200；遍历 38 个模型引用接口全部返回 200，
  其中 2 个模型合计返回 5 条真实引用。

旧知识索引已清空，不能视为已迁移到新知识链路；需要使用新结构重新导入/解析，
本次没有伪造文档 ID、回填猜测映射或自动调用模型重建索引。
此前备份仍可作为受控恢复的来源，但未演练恢复；本轮没有增加恢复副本。
既有跨项目文档/测试对齐差异及业务全链路验收不包含在上述结构验收结论中。

显式清理入口（默认只读预演，作用范围固定为本报告）：

```bash
# Ammolite 根目录，ammolite 环境
python scripts/deployment/init_postgresql.py --retire-reported-local-schema-only
# 执行需追加：--discard-reported-legacy-data --expected-plan-hash <本次预演哈希>

# KBot 根目录，kbot4 环境及工作区 PYTHONPATH
python -m scripts.db.retire_reported_local_columns
# 执行需追加：--discard-reported-fields --expected-plan-hash <本次预演哈希>
```

两个入口均不自动备份，只接受已确认的本地目标；规范重新使用的表、列或索引禁止删除。
PG 在单事务内锁定范围并复核计划/旧记录指纹，依赖超出删除清单时失败而不级联删除。
Oracle 每个表的旧列一次 DDL 删除，自动提交后立即比较所有保留规范列的内容指纹。

以下为前阶段无损修复与清理前盘点记录；其中“尚待确认”项目已经按上述结果处理，
不是当前残留差异。

## 结论与范围

前阶段完成可无损执行的结构修复时尚未达到全库零差异，以下保留当时的盘点与执行记录。
目标仅为本地 PostgreSQL `localhost:5432 / ammolite / ammolite` 和本地 Oracle
`127.0.0.1:1521 / kbot4 / KBOTDEV`，未操作 OCI 远端数据库。

规范取自当前工作树的 Entity、初始化入口、Oracle 分服务规范 DDL 和 manifest；
包含工作树中已经登记的数据保护在途结构，不覆盖其他人的在途修改。
PG 规范为 259 表、3781 列；Oracle 规范为 200 表，另有 14 个规范视图。
Oracle 列验收现已覆盖原先遗漏的 9 张 Table-only 通知投递表，而不只是 191 个映射类。

## 已实际应用

- PG：创建 24 张缺表（AIOps 独占知识链路 18 表、数据保护 6 表），包括规范索引和外键。
- PG：为 `aiops_knowledge_index` 增补可空的 `parse_run_id`、`index_generation_id`。
- PG：修复 14 处默认值。补足业务系统/草稿/资源图等表的 12 个时间列 `now()`；
  移除 `aiops_target.deployment_type` 的旧默认值和 `core_auth_session.entry` 的旧默认值。
  默认值修改仅影响后续写入，没有更新历史行。
- Oracle：从 `database/oracle/aiops_agent/011_ops_recovery.sql` 精确提取并执行
  6 张数据保护缺表及 8 个索引的声明，没有整份重放已有表。
- Oracle 源码：5 个授权标志列改为与既有规范 DDL 一致的 `CHAR(1)`；
  托管凭据 nonce 声明改为 `RAW(12)`。没有对既有数据库字段做类型转换。
- 修复 Oracle 验收工具对 CHAR、SMALLINT/Boolean 的处理，并补充通知表覆盖；
  PG 全库列验收现在也比较规范默认值。

全部新增表为空。重复运行结构增补和默认值预览均为无待执行语句。
模型、模型凭据、托管密钥、租户模型授权/配置及已采集账号配置的内容指纹保持不变。
PG 的两条旧知识索引记录完整保留，新增可空字段均为空。
没有 DROP/TRUNCATE、业务行回填、模型种子覆盖、服务重启、提交或推送。

## 尚待确认：PG

### 15 张规范外旧表

这些记录不是可直接删除的“空结构”，其中旧共享知识目录含 32 份文档记录。
不能把行数直接相加解释为独立文档数量。

| 旧表 | 行数 |
| --- | ---: |
| kc_bundle | 5 |
| kc_bundle_revision | 7 |
| kc_bundle_revision_document | 33 |
| kc_collection | 2 |
| kc_discovery_object | 14 |
| kc_document | 32 |
| kc_document_version | 32 |
| kc_evidence | 14 |
| kc_ingestion_job | 26 |
| kc_ingestion_receipt | 7 |
| kc_parse_view | 9 |
| kc_relation | 0 |
| kc_visual_asset | 0 |
| platform_case_resource | 1 |
| platform_resource_domain_collection | 2 |

### 12 个旧字段

- `aiops_knowledge_base.collection_id`：1 条非空。
- `aiops_knowledge_index.bundle_id / bundle_revision_id / collection_id`：各 2 条非空。
- `aiops_recovery_profile.backup_policy_enabled / backup_retention_days / backup_frequency_hours / backup_success_max_age_hours`：各 3 条非空。
- `knowledge_retrieval_agent_version.knowledge_core_id`：表为空。
- `platform_installation.deployment_mode`：1 条非空。
- `platform_license.deployment_mode`：0 条非空。
- `platform_module_catalog.supported_modes`：7 条非空。

### 3 个不能伪造的必填字段

`aiops_knowledge_index.document_id / document_version_id / ingestion_job_id` 在规范中均为 NOT NULL，
但现有 2 条记录仍引用旧 bundle，且新增 App 独占文档表为空。
因此未伪造 UUID、未降低非空约束，也未猜测跨架构历史映射。
需要明确真实归属后重新导入/解析，或在授权后废弃旧解析数据，再落实必填列。

### 5 个规范外普通索引（未删除）

- `aiops_database_resource.ix_aiops_database_resource_target_id`：额外访问索引。
- `aiops_recovery_drill.ix_aiops_recovery_drill_execution_target`。
- `aiops_recovery_drill.ix_aiops_recovery_drill_target_id`。
- `aiops_recovery_drill.ix_aiops_recovery_drill_resource`。
- `aiops_recovery_profile.ix_aiops_recovery_profile_resource`。

后四个与保留的规范索引重复。没有擅自删除结构；可与旧结构清理一并确认。

## 尚待确认：Oracle 的 8 个旧字段

- `KBOT_AI_MODEL.SUPPORTS_IMAGE_GENERATION / SUPPORTS_RESPONSES_STREAMING / SUPPORTS_X_SEARCH`：
  每列 30 条非空；`CAPABILITY_VERIFIED_AT` 为 1 条非空。模型表共 30 行，不允许删除模型记录。
- `KBOT_OPS_RECOVERY_PROFILE.BACKUP_POLICY_ENABLED / BACKUP_RETENTION_DAYS / BACKUP_FREQUENCY_HOURS / BACKUP_SUCCESS_MAX_AGE_HOURS`：
  每列 3 条非空。

上述字段均已备份，当前没有删除。清理字段不等于删除模型、API Key、账号或资产记录。

## 校验结果与边界

- 两仓架构边界检查通过。
- 40 个定向测试通过：PG 部署/验收单元与初始化合同 13 个、既有知识配置部署测试 3 个，
  Oracle 部署/列验收及账号访问测试 24 个。
- 两仓 App 独占知识固定 36 表离线检查通过；PG AIOps metadata 112 表检查通过。
- Oracle 规范 DDL 静态验收通过：8 服务、39 脚本。
- Oracle Catalog 实库验收通过：214 对象。Entity 实库验收仍因上述 8 个旧字段失败。
- PG Catalog/Entity 实库验收仍因上述 15 张旧表、12 个旧字段、3 个必填缺列失败；
  没有现存规范列的类型、空值、主键或默认值漂移。
- 只读扩展核对：PG 规范主键/外键/索引均存在，定义覆盖未见偏差，索引均有效，
  外键均已验证；保留上述 5 个额外索引。
- Oracle 200 表主键、404 个规范外键及 479 个规范显式索引未见缺失或实际定义偏差；
  约束均已验证，索引无不可用项。时区索引的 SYS_EXTRACT_UTC、CASE 表达式优化
  已按实际列类型及等价分支核对，不作为结构错误；LOB 内部索引不作为额外业务索引。
- Oracle 2823 个规范建表列的默认值以及 2 个 identity 声明核对一致；
  ALTER 后增列不在这项默认值统计内。
- 两仓 AIOps 跨项目检查另有既有不一致，未归因于本次 Schema 修改：
  通知投递说明、通知投递应用测试、daily baseline 测试的字节/AST 对齐，共 5 条失败。
- 本次不等于所有业务链路、视图定义或所有应用约束策略均已验收通过。

## 备份与恢复注意

本机持久目录：
`/home/chris/.local/state/ammolite/schema-alignment/20261117-mYxu2i/`

目录 0700，备份/清单 0600；不得提交到 Git 或输出其业务行内容。

- `pg-schema-before.dump`：整个目标 Schema 的结构快照。
- `pg-affected-data-before.dump`：受影响旧表/配置及模型凭据等定向数据备份，
  **不是包含数百万运行行的全库数据备份**。
- `oracle-schema-before.json`：194 张既有 KBOT 表的 DBMS_METADATA 建表快照。
- `oracle-protected-data-before.json`：模型、托管凭据、恢复配置、账号及凭据等 45 行受保护数据。
  二进制、LOB、时间、Decimal 保留带类型的序列化表示。
- 其他 JSON 为结构盘点、预览计划、执行计划、指纹与扩展核对证据。
- `SHA256SUMS` 中四份关键备份校验通过；PG custom dump 目录可读取。
- PG 数据备份含循环外键依赖，恢复时需要按依赖设计受控恢复流程，不能在运行库盲目回灌。
  Oracle DDL 自动提交，逐句执行失败时停止，不以 DROP 已有表“回滚”。
  当前未实际演练恢复。

## 可重复的显式部署入口

在 Ammolite 根目录、ammolite 环境中，只读预览：

```bash
python scripts/deployment/init_postgresql.py --align-schema-additions-only
python scripts/deployment/init_postgresql.py --align-schema-defaults-only
```

检查输出 target、缺失对象、blocked_columns 和 DDL 后，在同一命令追加
`--expected-plan-hash <本次预览哈希>` 才执行。两种模式互斥，均不调用全量初始化。

在 KBot 根目录、kbot4 环境并配置工作区 PYTHONPATH 后：

```bash
python -m scripts.db.align_oracle_schema_additions
```

默认只读预览；追加 `--expected-plan-hash <本次预览哈希>` 才逐句增补缺表及其附属声明。
该入口不修复已有表缺列或不完整的既有索引。中途失败应先重新盘点，再决定恢复动作。
不得用空库初始化器代替现库对齐，不得在启动、readiness、UoW 或业务请求中执行 Schema 探测/DDL。

原待办中的旧结构及对应旧数据已经按用户确认的范围清理，当前结构验收结果以文首为准。
原始知识资产保留，但旧解析结果不再可用；重新导入/解析属于后续知识数据准备。

