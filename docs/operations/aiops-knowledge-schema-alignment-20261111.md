# AIOps 知识结构对齐与旧解析清理

## 故障与用户批准范围

知识库列表修复后，资料版本详情仍返回 503。2026-11-11 09:22:11 的在线日志显示：

```text
ORA-00904: "KBOT_OPS_KNOWLEDGE_INDEX"."INDEX_GENERATION_ID": invalid identifier
```

索引表仍是 Collection/Bundle 结构，缺少当前规范中的 5 个文档/任务/解析/代次字段。
另外 18 张 App 自有知识表和案例配置表尚未创建。用户最终明确要求：
**只做结构对齐，删除旧的不兼容解析数据，由用户重新上传，不做备份。**

此次不迁移旧资料，不重新入库或解析，不选择模型，不改业务运行时代码、不重启服务；
不写共享 KC、知识检索 App、账号、模型、Prompt 或其他 AIOps 业务表，不删除源文件。

## 执行工具

在仓库根目录运行：

```bash
python scripts/deployment/align_aiops_knowledge_schema.py
```

默认只读。仅加载当前清单校验通过的 `010_ops_knowledge.sql` 和
`014_ops_knowledge_ingestion.sql`，范围固定为 28 张 AIOps 知识相关表。
元数据校验只属于显式部署工具，不注册到启动、readiness 或业务请求中。

写入必须同时提供：

```text
--apply --discard-legacy-parses --expected-plan-hash <本次只读预演指纹>
```

执行前只保留内存中的结构、数量和不可逆数据指纹，不生成快照文件或业务内容备份。
变更计划包括：

- 创建缺少的 18 张自有知识表及配置表，补齐规范访问索引和循环外键。
- 锁定旧知识资产/版本及其附属表，复查待删除版本 ID，按资料行版本进行条件更新。
- 仅删除旧索引定位到的解析版本、范围、来源、索引和审核记录。
- 资料条目保留，清空当前版本指针、恢复 DRAFT 并递增行版本；知识库及 Agent 绑定保持不变。
- 删除知识库旧 `COLLECTION_ID` 列，以及索引表旧 `COLLECTION_ID/BUNDLE_ID/BUNDLE_REVISION_ID` 列；
  空索引表新增规范文档及任务字段，保持规范非空要求，绝不伪造文档 ID。

结构校验对比列名、类型、字符/字节长度语义及可空性；遇到非批准差异拒绝执行。
删除按绑定版本 ID 限定，数量必须与预演一致；不提供任意表名、SQL 或连接串入口。
新表使用完整规范 DDL，现有索引按规范名称补齐；三条后置循环外键同时检查是否已启用。

Oracle DDL 与已提交清理不可整体回滚。工具失败时输出已完成语句名称/删除数量，
不声称已经回滚；中断后的空索引表允许继续补齐缺失字段。完成状态重复预演为空计划。

## 线上执行结果

通过用户指定 SSH 密钥和 `curve25519-sha256` 连接 kbotdev，只使用 KBot 配置访问其 Oracle。
执行目标指纹与故障诊断时一致；计划指纹为：

```text
82aebe2fb91926133f9a263ee13e54a976cd0421e99c8814d70df8ffc1358155
```

完成 87 条缺失结构创建语句及 3 条旧列对齐语句，28 张限定表复核通过。
再次只读预演：`create_statements=0`、`alter_statements=[]`、`discard_versions=0`。

已删除旧解析记录：版本 2、范围 9、来源 11、索引 2、审核 9，共 33 条。
本次没有备份，不能从本次操作恢复这些已删除记录。

保留：知识库 2 个、资料条目 2 个、Agent 版本知识库绑定 25 条。
知识库及绑定逐值指纹保持不变；资料 ID、归属、名称、安全级别等非清理字段保持不变。
两个资料条目均为 DRAFT、`current_version_id=null`、`versions=[]`，等待用户重新上传。
共享 KC 历史表和源文件未清理；不把其他 App 的历史解析视为本次删除目标。

## 在线 API 验证

使用服务器内存中的 60 秒短期已授权管理令牌执行只读请求，不输出或保存令牌、用户信息。
通过用户指定在线开发日志 API 按 Request ID 复核：

```text
ops-aiops-knowledge-align-20261111-e19a2a9d-6bc8-4b7a-bb46-b57a2d67d021
```

13 个请求返回 200：知识库列表（含归档/不含归档）、两个知识库详情及模型配置、
知识概览、标签、案例列表、案例模型配置、资料列表和两个资料详情。
标签仍有 133 项，案例列表为空，资料条目仍为 2 项。

已删除的旧版本 `01a15841-573a-7a50-8922-f643b2149871` 返回预期的
404 / `AIOPS_KNOWLEDGE_ASSET_NOT_FOUND`，不再是数据库异常引发的 503。

本次不上传或解析验证文档，不声称完成上传、解析、发布、检索的端到端验收。
当前知识库配置仍为空，线上表示空间数量为 0；用户重新上传前还需显式配置模型及与之
匹配的表示空间/Profile。这是配置前提，不由结构对齐工具猜测或自动选择。

## 本地验证

```bash
conda run -n kbot4 python tests/contract/test_aiops_knowledge_schema_alignment.py
conda run -n kbot4 python tests/acceptance/check_4_0_boundaries.py
conda run -n kbot4 python tests/acceptance/check_oracle_schema.py
PYTHONPATH=packages/knowledge_algorithms/src conda run -n kbot4 python -m unittest tests.unit.aiops_agent.test_aiops_operations_knowledge tests.unit.main_api.test_aiops_app_operations_knowledge
```

10 项部署工具合同、21 项 KBot 知识合同/API、架构边界以及 Oracle 规范 DDL 检查通过。
此次为 Oracle 专用结构修复，不改变跨项目产品合同；另在 Ammolite 仓库验证同一知识能力的
21 项测试全部通过，没有修改其数据库或代码：

```bash
PYTHONPATH=packages/knowledge_algorithms/src conda run -n cube python -m unittest tests.unit.aiops_agent.test_aiops_operations_knowledge tests.unit.main_api.test_main_api_aiops_operations_knowledge
```
