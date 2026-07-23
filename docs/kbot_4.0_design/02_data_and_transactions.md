# 4.0 数据访问、事务与任务规范

## 当前问题

现有代码多数 Repository 已通过构造函数接收 `AsyncSession`，这是正确基础；但 Service 常在每个方法内自行打开 Session，跨 Service 用例会变成多个事务。部分旧 Repository 还调用 `commit()`，使调用方无法控制原子性。4.0 禁止在新代码中复制这两种做法。

典型风险是“删除文件 → 删除图谱 → 删除知识库”分别提交：中途失败后会留下半完成状态。知识库入库也不能把“创建版本、投递解析任务、更新状态”拆成互不关联的提交。

## 分层责任

```text
Route / Worker
  → Application Service（定义一个业务用例）
    → Unit of Work（一个 Session、一个事务、领域 Repository 集合）
      → Repository（SQL、Entity 映射、flush）
```

- Route 只做鉴权、DTO 校验和错误映射；不得直接构造 Repository。
- Application Service 组合领域操作、决定事务范围，并在用例全部成功后显式调用 `uow.commit()`。
- Unit of Work（UoW）创建/管理 Session、执行显式提交；异常或未提交退出时回滚并关闭。
- Repository 只能接收 `AsyncSession`，不得创建 Session、不得 `commit()`/`rollback()`；需要数据库生成 ID 时可 `flush()`。
- Entity 是持久化模型，不跨 API 返回；跨层传递 Command、Result 或领域对象。

4.0 领域实体统一使用 UUIDv7：Oracle 以 `RAW(16)` 保存 PK/FK，未来 PostgreSQL 版本使用原生 `uuid`。API 和跨进程契约序列化为规范 UUID 字符串，不再维护数字 PK 与 Public UID 的双层映射。`APP_ID`、`DOMAIN_ID`、版本、序号和计数仍使用数值，详见 [31_aiops_step2_persistence_and_identity.md](31_aiops_step2_persistence_and_identity.md)。

## UoW 参考接口

```python
class KnowledgeUnitOfWork:
    def __init__(self, session: AsyncSession):
        self.collections = CollectionRepository(session)
        self.bundles = BundleRepository(session)
        self.documents = DocumentRepository(session)
        self.jobs = IngestionJobRepository(session)
        self.outbox = OutboxRepository(session)

async with knowledge_uow_factory() as uow:
    await ingestion_service.ingest(command, uow)
    await uow.commit()
# 异常或漏掉显式 commit 时，UoW 回滚并关闭
```

HTTP 请求、CLI 和 Worker 使用同一个 UoW factory；测试可以注入事务型测试 UoW 或 Fake Repository。不要把 FastAPI 的 `Depends` 对象渗透到 domain 层。

采用显式提交是为了让“读完后提前返回”“策略拒绝”“状态冲突”等分支默认不写入。Repository 永远不能提交；Application Service 只能通过 UoW 提交，不能直接操作 Session。

Agent Runtime 使用独立的 `AgentUnitOfWork` 管理 Run、Task、Artifact 和 Event
Repository。Task 完成、Artifact 写入、状态迁移和后继 Task 就绪必须在同一短事务
内完成；模型调用、KC HTTP 调用和外部副作用均在事务之外执行。事件写入与状态
写入使用同一提交，避免 SSE 观察到不存在的 Artifact 或已经完成但没有事件的 Task。

## 事务与 Outbox

一个事务只覆盖同一数据库内必须原子完成的状态变更。例如 Bundle 入库须在同一提交中写入 Bundle、Document、Document Version、`KBOT_KC_INGESTION_JOB` 和 Outbox 事件。提交成功后，Dispatcher 才将任务交给 Parser/Indexer。

禁止在未提交事务内调用 Parser、Embedding、LLM 或任意 HTTP 服务：远端成功但本地回滚、或反向情况都会造成不可恢复的不一致。远端操作必须由可重试任务驱动，并以幂等键（`job_id`、`document_version_id`、`content_hash`）保护。

## Job 语义

`KBOT_KC_INGESTION_JOB` 是持久化队列，不是普通日志。它至少包含状态、尝试次数、租约持有者、租约到期、下一次执行时间、参数快照、错误摘要和幂等键。

- Worker 以条件更新领取租约，避免多个副本重复处理。
- 成功时写结果并推进后续 Job；失败时按可恢复/不可恢复分类重试或终止。
- Worker 崩溃后，过期租约可重新领取；不得依赖进程内 `set()` 保证全局去重。
- 每个 Job 在短事务中领取，在外部计算后用新事务写回结果；不得长时间持有数据库连接。

## 迁移规则

新代码立即遵守上述规范。旧 Repository 内的 `commit()` 需先改为 `flush()`，由调用 Service/UoW 提交；每次修改都增加“成功、异常回滚、跨 Repository 原子性”测试。禁止大规模机械替换而不验证调用方事务边界。
