# 4.0 资源标识与 AIOps Persistence 设计

## 当前现状

当前 3.5 Knowledge Core 的 `COLLECTION_ID/BUNDLE_ID/DOCUMENT_ID/DOCUMENT_VERSION_ID/EVIDENCE_ID` 等均为 Oracle `NUMBER(38) IDENTITY`，Python Entity 和 API 也使用 `int`。旧 Agent 表和过渡 `agent/common` 同样使用整数 Agent ID。

这对单库内部连接有效，但不适合作为 4.0 跨服务、外部系统和未来 PostgreSQL 版本的稳定身份：整数拆库后不再全局唯一，也会把数据库分配策略泄漏为长期契约。4.0 不承担兼容包袱，因此直接统一身份，避免长期维护“数字 PK + Public UID”的映射层。

## 标识决策

每个可独立寻址的领域资源只使用一个 UUIDv7 主键，不设置同义的数字 PK 或 `*_UID`：

| 层次 | Python/契约 | Oracle | PostgreSQL |
| --- | --- | --- | --- |
| 领域对象、API、事件 | `uuid.UUID` / UUID 字符串 | `RAW(16)` | `uuid` |
| 同领域 PK/FK | `uuid.UUID` | `RAW(16)` | `uuid` |
| 纯关联/无独立生命周期子项 | 复合键 | UUID FK 组合 | UUID FK 组合 |
| 序号、版本、计数 | `int` | `NUMBER` | `bigint/integer` |

字段保持自然名称：数据库与 API 均使用 `AGENT_ID`、`COLLECTION_ID`、`DOCUMENT_ID`，不再并存 `COLLECTION_ID/COLLECTION_UID`。纯关联表使用两端 ID（必要时加角色/类型）作为复合主键，不额外创建无业务含义的 ID。`APP_ID`、`DOMAIN_ID` 是现有 APEX/租户字段，仍为 `NUMBER(38)`；`VERSION_NO`、`ORDINAL`、`SEQUENCE_NO`、`ROW_VERSION` 等非身份字段也继续使用数值。

UUID 由 Application Service 在聚合创建前生成，使用 UUIDv7 以获得时间有序写入；规范格式为小写 `8-4-4-4-12`。数据库只做唯一性和长度约束，格式/版本由应用验证。UUID 不是授权凭据，所有读取仍校验 Domain、Binding 和 Security Level。

生成和解析统一封装为平台 Identity 工具；Domain、Application、Pydantic DTO 和 Repository 接口均使用 `uuid.UUID`。Oracle Adapter 只在 Entity TypeDecorator 中转换 `uuid.UUID ↔ RAW(16)`，PostgreSQL Adapter 使用原生 UUID 类型。Python 版本或 UUIDv7 库差异不得渗透到业务代码。

### 为什么不使用双层 ID

双层 ID 适用于兼容旧库、不能重写高频 Join，或必须隐藏既有序号的系统。本项目处于 4.0 断代重构期，继续双层会增加 UID→PK 查询、DTO/Mapper 分支、唯一索引、排障歧义和跨数据库迁移成本，没有对应收益。

UUID 不以 `VARCHAR2(36)` 参与内部 Join。Oracle 使用 16 字节 `RAW(16)`，PG 使用原生 16 字节 `uuid`；UUIDv7 的时间有序特性也比随机 UUID 更有利于 B-tree 写入局部性。APEX 通过业务视图把 RAW 格式化为规范 UUID 字符串，页面不直接处理二进制值。

## Agent ID

4.0 Agent 领域身份定义为：

```text
AGENT_ID RAW(16) PRIMARY KEY   # UUIDv7
```

Agent 不维护数字主键或额外 UID。KC 的 Collection Binding、AIOps 的 Target Binding、通用 Agent Run/Delegation 和 `PARENT_AGENT_RUN_ID/PARENT_DELEGATION_ID` 均使用同一 UUID 类型。

外部系统接入时获得的是稳定 `agent_id`，而不是 Agent Name。Agent Name/Key 可修改或按 Domain 重复，不能作为身份；外部系统也不能自行提交任意 UUID扩大权限，Main API 必须验证该 Agent 已授权给当前 Domain/主体。

若未来接入由其他平台管理的 Agent，KBot 仍生成自己的 `agent_id`，另存 `(source_system, external_agent_id)` 映射；不能把第三方命名空间直接塞进本地 UUID 字段。

## Knowledge Core 与 AIOps ID

KC 的 Collection、Receipt、Bundle、Bundle Revision、Document、Document Version、Parse View、Ingestion Job、Evidence 和可单独撤销/审计的 Collection Binding，直接以 UUIDv7 `*_ID RAW(16)` 为 PK/FK。Revision Document 等真正无独立生命周期的关联表使用外键组合为主键；投影是否保留独立 ID 由其是否需要单独重建、租约或审计决定。`COLLECTION_KEY`、`SOURCE_ID`、`EXTERNAL_DOCUMENT_ID`、`EVIDENCE_KEY` 仍是自然键、来源键或作用域内稳定键，不替代主键。

AIOps 的 Target、Policy、Monitor Source、Event、Alert、Run、Task、Artifact、Proposal、HITL、Execution、Inspection Plan、Inspection Fire、Report、Inbox 和 Outbox 等独立实体也使用 UUIDv7 `*_ID RAW(16)`。具有配置、状态、审计和独立 API 生命周期的 Target Binding、Target Monitor、Inspection Target 同样使用单一 UUIDv7 主键；Run Event 使用 `(OPS_RUN_ID, SEQUENCE_NO)`。Task 的 `ORDINAL` 等仅用于聚合内排序，不是实体身份。

API 中的 `target_id/ops_run_id/proposal_id/hitl_id/report_id` 等字段是同一主键的规范 UUID 字符串表示；无需先解析 UID 再查询内部数字 PK。

## 迁移与兼容边界

- 4.0 不接受旧整数 Agent ID；旧 Agent 配置重建时生成新 UUID，并输出一次性映射报告；
- 4.0 KC API 不接受数值 Collection/Bundle/Document ID；Portal/APEX/Agent Client 同步改用 UUID；
- 开发库通过重建或一次性迁移生成 UUID，不做长期双写，也不把旧数字嵌入 UUID；
- UUID 不允许调用方在 Create 时指定，除非是受信离线导入且携带专用 Scope；
- 外部来源自己的 ID 保存在 `SOURCE_ID/EXTERNAL_DOCUMENT_ID`，不能直接作为 KC 主键；
- APEX 查询视图使用 `RAW_TO_UUID(<ID>)` 输出规范字符串；写入与筛选优先经 API，确需直连时使用受控转换函数，不在页面复制转换规则；
- PG 适配版只替换数据库类型和 SQL 方言，API、领域对象、ID 值及跨服务契约保持不变。

## AIOps 聚合与 Repository

步骤 2 按聚合而非“一表一个 Service”组织 Repository：

```text
TargetRepository            Target、Agent Binding、Target Monitor
MonitorSourceRepository     Monitor Source、健康状态
PolicyRepository            不可变策略版本和 Active 查询
AlertRepository             Event、Alert 关联
OpsRunRepository            Run、Task、Artifact、Run Event
ChangeRepository            Proposal、HITL、Approval Token、Execution
InspectionRepository        Plan、Plan Target、Fire、Report
InboxRepository             外部消息去重
OutboxRepository            可靠交付、租约和重试
```

Repository 方法直接接收 UUID，并在同一查询中校验 Scope；不提供 `get_pk_by_uid` 中转查询。Repository 不创建 Session、不调用 `commit/rollback`、不访问 HTTP/LLM/Monitor/Secret Store。

Root 资源查询必须带 `app_id/domain_id`；子资源通过 Join 到 Target/Plan 验证 Scope。UUID 只能降低碰撞和枚举风险，不能替代授权。

## Entity 与 Domain Mapping

SQLAlchemy Entity 位于 `aiops_agent/entities`，逐列映射 [30_aiops_step1_oracle_schema.md](30_aiops_step1_oracle_schema.md)：

```text
NUMBER(38)       → Mapped[int] + Numeric(38, 0)
NUMBER(19)       → Mapped[int] + Numeric(19, 0)
UUID RAW(16)     → Mapped[uuid.UUID] + UUIDv7Raw
JSON CLOB        → OracleJSON
TIMESTAMP TZ     → DateTime(timezone=True)
```

Entity 不继承 Domain Aggregate，不实现状态迁移，不使用跨表 lazy relationship。Application Mapper 显式构建 Domain Snapshot/Command；状态值进入 Domain Enum 后再持久化。ID 使用专用 `ResourceId` Value Object 校验并规范化，不能在业务代码散落 `str(uuid)`。

建议实体文件：

```text
entities/target.py
entities/monitoring.py
entities/runtime.py
entities/change.py
entities/inspection.py
entities/messaging.py
```

## AIOpsUnitOfWork

```python
class AIOpsUnitOfWork:
    targets: TargetRepository
    monitor_sources: MonitorSourceRepository
    policies: PolicyRepository
    alerts: AlertRepository
    runs: OpsRunRepository
    changes: ChangeRepository
    inspections: InspectionRepository
    inbox: InboxRepository
    outbox: OutboxRepository

    async def __aenter__(self): ...
    async def commit(self): ...
    async def rollback(self): ...
    async def __aexit__(self, exc_type, exc, tb): ...
```

UoW Factory 接受 App 自己的 `async_sessionmaker`。进入时创建短生命周期 Session/Transaction；正常退出也不隐式提交，Application Service 必须显式 `commit()`，异常统一回滚并关闭。这样漏写 Commit 会安全失败，而不会悄悄持久化半成品。

一个 UoW 只允许提交一次；`commit()` 必须是用例事务的最后一个数据库动作。提交后继续写 Repository 应抛出错误，不能悄悄开启第二个事务。需要后续读取或写入时创建新的 UoW。

同一 AIOps 用例可以在一个 UoW 修改多个聚合，例如 Webhook 同事务写 Inbox/Event/Alert/Run/Task/Outbox；这仍是单一领域事务，不是跨服务直接写表。任何 Monitor、LLM、KC、Executor 或 Secret Store 调用必须在事务外。

## 乐观锁与状态命令

所有可变聚合使用条件更新：

```sql
UPDATE ...
   SET STATUS = :new_status,
       ROW_VERSION = ROW_VERSION + 1,
       UPDATED_AT = SYSTIMESTAMP
 WHERE <PK> = :pk
   AND ROW_VERSION = :expected_version
   AND STATUS IN (<allowed_sources>)
```

影响零行映射为 `OPS_ROW_VERSION_CHANGED` 或 `OPS_STATE_CONFLICT`，Application Service 重新读取后决定，不做盲重试。不可变 Artifact/Event/Policy 内容只 Insert；状态迁移和内容版本分离。

## Task、Plan 与 Outbox 租约

领取使用 Oracle `FOR UPDATE SKIP LOCKED`，事务只完成选取和写租约：

1. 按 `STATUS/AVAILABLE_AT/PRIORITY` 选择一小批 Ready 行；
2. 锁定并写 `RUNNING/PUBLISHING`、Lease Owner/Until、Attempt 和 Row Version；
3. Commit 后执行外部调用；
4. 完成时校验 PK、Lease Owner、每次领取生成的 Lease Token 和 Lease 未过期；
5. 旧 Worker 的迟到结果返回 `STALE_LEASE`，不得写 Artifact。

过期 `RUNNING` 不由普通 Claim 偷取。Reconciler 单独把它转换为 `RETRY_WAIT/FAILED/UNKNOWN`，清除旧 Lease Token 并写 Run Event 后才可再次领取。Task、Plan 和 Outbox 每次领取都生成新的 `LEASE_TOKEN RAW(16)`；迟到写回必须同时匹配 Owner、Token 和有效期。Scheduler 对 Plan 使用同样原则，领取 Plan 后在一个事务内创建 Run/Initial Task/Outbox 并推进 `NEXT_RUN_AT`。

## Run Event 序号

写 Run Event 时先 `SELECT Run FOR UPDATE`，再通过 `(OPS_RUN_ID, SEQUENCE_NO)` 索引读取当前最大序号并插入 `max + 1`。Run 锁使同一 Run 序号严格单调；不同 Run 可并行。序号从 1 开始，只用于该 Run 的 SSE Cursor，不作为全局资源 ID。

如果压测证明 Run 行竞争明显，再单独增加 Sequence Allocator 表；首版不为假设性吞吐增加第 21 张领域表。

## Inbox/Outbox 事务语义

- Webhook/Executor Callback 先按 `(source_system, message_key)` Insert Inbox；唯一冲突读取原处理结果并返回，不重复推进领域状态；
- Application Service 在同一事务写领域变化和 Outbox；
- Dispatcher 只在 Commit 后投递，成功后条件更新 `PUBLISHED`；
- 消费者按业务幂等键处理，因此允许 Outbox 至少一次投递；
- Payload 大于阈值时先在事务外写对象存储，再在事务内写 URI/Hash；数据库失败后的孤立对象由 Cleanup Job 清理；
- Mutation 回调乱序时按 `status_version` 只接受向前迁移，`UNKNOWN` 不自动覆盖已确认终态。

## 事务边界示例

Worker 执行诊断 Task：

```text
UoW-A: claim Task + lease + event → commit
外部: Monitor/DB Executor/LLM/KC
UoW-B: validate lease/version + insert Artifact + finish Task
       + ready successor + update Run + RunEvent/Outbox → commit
```

审批命令：

```text
UoW: scope/assignee/version/policy recheck
   → approve HITL/Proposal
   → store Token Hash
   → create Execution/Outbox/Event
   → commit
```

任何外部调用失败都不会占用长数据库事务；恢复依赖已提交 Task/Artifact/Inbox/Outbox，不依赖进程内对象。

## 步骤 2 文件布局

```text
aiops_agent/
  domain/*/entities.py, value_objects.py, state_machine.py
  entities/{target,monitoring,runtime,change,inspection,messaging}.py
  repositories/{target,monitoring,runtime,change,inspection,messaging}.py
  persistence/uow.py
  persistence/mappers.py
  application/errors.py
  tests/fixtures/uow.py
```

Repository Protocol 放在 Application Port，Oracle 实现放在 `repositories`。测试 Fake 实现同一 Protocol，不允许为测试向生产 Repository 添加绕过 Scope/Lock 的方法。

## 完成定义

- Agent API/数据库/Binding 全部使用 UUIDv7，不再接受整数 Agent ID；
- KC/AIOps 的领域 PK/FK、API 和跨进程契约使用同一个 UUIDv7，不存在双层 ID；
- Oracle 使用 `RAW(16)`，PostgreSQL 使用原生 `uuid`，数据库差异仅存在于 Persistence Adapter；
- Entity 与 Oracle Manifest 逐列一致，Domain 不依赖 SQLAlchemy；
- Repository 无 Commit 和外部 I/O，所有写入由显式 UoW 控制；
- 乐观锁、租约、Run Event、Inbox/Outbox 在崩溃、重放和并发下不重复产生业务结果；
- DB Executor 仅通过 UUID 请求和 Claim 契约工作，不获得 KBot Schema Session。
