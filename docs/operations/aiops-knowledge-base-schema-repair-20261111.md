# 运维知识库列表 503 定向修复

## 故障与范围

2026-11-11 09:10:02（北京时间），在线开发日志记录了知识库列表请求失败：

```text
GET /api/v1/apps/aiops/operations-knowledge/knowledge-bases -> 503
aiops_agent -> ORA-00904: "KBOT_OPS_KNOWLEDGE_BASE"."CONFIGURATION_JSON": invalid identifier
```

AIOps 返回的内部 500 被 Main API 规范化为外部 503。线上代码与本地基线均为
`31d59502`；线上 Oracle 表仍保留旧 `COLLECTION_ID NOT NULL`，缺少当前规范要求的
`CONFIGURATION_JSON JSON NOT NULL`。同机 Ammolite 的进程及数据库不在此次变更范围。

只修复 `KBOT_OPS_KNOWLEDGE_BASE`，不运行全库初始化、不更新 Schema 版本视图、
不迁移 KC 文档或集合、不修改服务运行时代码、不重启服务。

## 工具与安全边界

从仓库根目录、使用已安装工作区依赖的 Python 环境执行：

```bash
python scripts/deployment/repair_aiops_knowledge_base.py
```

默认只读，输出数据库目标指纹、固定 SQL 计划、计划指纹和记录数量。只有显式提供
`--apply --expected-plan-hash <只读计划指纹> --report-dir <现有0700目录>` 才执行变更。
通过共享配置访问器获取连接信息，不输出用户名、口令、完整连接串或业务内容。

固定计划仅允许：

1. 缺少配置列时增加原生 JSON 列。
2. 只将 NULL 配置初始化为 `{}`，不覆盖已有配置、不猜测模型或 Profile。
3. 将配置列收紧为 NOT NULL。
4. 旧 `COLLECTION_ID` 仍为必填时解除其 NOT NULL，原有值逐值保留；否则当前代码
   创建知识库时仍会被未使用的旧列阻断。新规范及运行时代码均不使用此列，不增加旧 KC 兼容路径。

新增列、半完成的可空列、已完成结构均有独立计划，重复执行完成态为空计划。
出现其他缺失字段、未知字段或不兼容列类型时拒绝执行，不自动扩大修复范围。

执行前保存知识库及资料/Agent 绑定的私有快照；执行后比较所有原有字段及关联，
包括旧 Collection ID、版本号、时间戳和已有非空模型配置。快照文件为 0600，禁止覆盖。
Oracle DDL 会自动提交，失败时保留已完成步骤，不能宣称整体事务回滚。

工具使用显式列投影并关闭语句缓存，避免 DDL 前后的列元数据复用。首次线上执行
四条 SQL 均已完成，但变更后快照遭遇驱动 `InternalError`；没有重复执行 DDL，
随后使用独立连接与原始快照完成逐值复核，保留原 `partial.json` 并写入最终成功记录。

## 线上结果

服务器：`kbotdev`；连接使用指定 SSH 密钥和 `curve25519-sha256`，未保存密钥到仓库。

审计目录：`/home/ubuntu/kbot-aiops-knowledge-repair-20261111-0hdAQ0`，包含
`before.json`、`plan.json`、`partial.json`、`after.json`、`completed.json` 和工具副本。
数据库目标仅记录不可逆指纹。

独立连接复核：

- 知识库 2 条，资料 2 条，Agent 版本绑定 25 条；所有原有字段与绑定保持不变。
- 配置列为原生 JSON NOT NULL；旧 Collection 列可空且原值未改变；剩余修复计划为空。
- 新增配置为 `{}`，不自动继承旧 KC 模型配置，模型需用户显式配置。

09:19:25—09:19:28，使用服务器内存中的 60 秒短期令牌，以现有且已授权的管理主体执行
6 次只读公开 API 请求；不输出或保存令牌及用户信息。下列请求均返回 HTTP 200：

- 知识库列表及 `include_archived=true` 列表，各返回 2 项。
- 两个知识库分别获取详情、模型配置。

在线日志入口：`http://140.238.44.208:18099/api/v1/development/logs/services`。
通过同一服务的 `/events` API，按下列 Request ID 已核对全部 6 条 Main API ACCESS 200：

```text
ops-aiops-knowledge-repair-20261111-e543341b-1255-44d4-84f4-3a51d54c8010
```

验证的是本次知识库列表、详情和配置读取，不代表完整 App-owned Knowledge Schema
升级、旧资料检索或上传链路已验收；只读检查还发现 `KBOT_OPS_KNOWLEDGE_CONFIG`
尚不存在，本次未借机创建其他功能表。

## 回归

```bash
conda run -n kbot4 python tests/contract/test_aiops_knowledge_base_repair.py
PYTHONPATH=packages/knowledge_algorithms/src conda run -n kbot4 python -m unittest tests.unit.aiops_agent.test_aiops_operations_knowledge tests.unit.main_api.test_aiops_app_operations_knowledge
conda run -n kbot4 python tests/acceptance/check_4_0_boundaries.py
```

定向工具 8 项测试、KBot 知识合同/API 21 项测试及架构边界检查通过。此次不改变领域行为、
接口或规范持久化模型；Ammolite 已具备相同配置字段，无需 Oracle 专用修复。已在 Ammolite
仓库分别验证知识合同/API 的 21 项测试，全部通过：

```bash
PYTHONPATH=packages/knowledge_algorithms/src conda run -n cube python -m unittest tests.unit.aiops_agent.test_aiops_operations_knowledge tests.unit.main_api.test_main_api_aiops_operations_knowledge
```
