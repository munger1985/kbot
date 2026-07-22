# 步骤 2 详细设计：KC 应用分层与事务边界

## 结论

Repository 接收 `AsyncSession` 是合理的；不合理的是让每个通用 Service 自行从全局 `get_session()` 开启、自动提交事务。KC 采用“Application Service 拥有 Use Case 和 Unit of Work，Repository 只使用同一 Session”的模式。

```text
FastAPI Controller
  → Command / Query DTO + Auth Context
  → KC Application Service（业务不变量、事务边界、编排）
  → Unit of Work（一个短生命周期 AsyncSession / transaction）
  → Repository（只读写 Entity，不 commit、不调外部服务）
  → Oracle
```

Parser、对象存储、模型和 HTTP Client 是 Application Service 调用的端口（port）；它们既不接受 Session，也不能直接调用 Repository。

## 为什么不沿用当前模式

现有 V1 中常见 Service 属性返回 `get_session()`、方法内部 `async with` 的模式，且 `core/database/oracle.py:get_session()` 在上下文退出时自动 commit。这能简化单表操作，但 KC 的一个 Use Case 往往同时改变 Bundle、Revision、Member、Version、Parse View、Job 与 Receipt；若内部调用再各自开 Session，就无法保证原子切换，也难以测试或防止局部提交。

KC 不修改 V1 的全局 Session 行为。新进程使用独立的 `KnowledgeCoreUnitOfWork`：默认 rollback，只有 Application Service 在满足全部数据库不变量后显式 `commit()`；读取 Query 不创建写事务。`flush()` 仅用于获得 ID 或执行约束校验，不能替代 commit。

## 建议模块布局

```text
knowledge_core/
  api/                 # V2 public/internal controllers、Pydantic DTO、依赖注入
  application/         # command/query handlers、状态机、use cases
  domain/              # 状态枚举、不变量、纯规则和领域异常
  persistence/         # KC entities、repositories、UnitOfWork、DDL migration
  ports/               # ObjectStore、ParserTask、Clock、Audit 等接口
  adapters/            # Oracle、对象存储、Parser HTTP、认证实现
```

可保留仓库现有 `dao/entities`、`dao/repositories` 作为物理目录，但 KC 文件必须以 `kc_` 前缀或独立子包隔离；不得把 V2 Repository 混入 `FileRepository`、`TxtChunkRepository` 或 V1 Service。

## Unit of Work 与 Repository 规则

| 组件 | 可以做 | 禁止做 |
| --- | --- | --- |
| Controller | 认证、DTO 解析、调用 Application Service、HTTP 映射 | 编排多表写入、直接构造 Repository、手动 commit |
| Application Service | Scope 校验后的 Use Case、状态转换、调用多个 Repository、显式事务提交、发布后补偿决策 | 拼接 SQL、跨过 UoW 提交、把 Session 传给外部服务 |
| Repository | 使用注入 Session 的查询/持久化、加锁查询、`flush` | `commit/rollback`、权限决策、状态机、对象存储/HTTP 调用 |
| Unit of Work | 创建/关闭 Session、begin/commit/rollback、暴露同一组 Repository | 业务规则、对外 HTTP 调用 |

示意接口：

```python
async with kc_uow_factory() as uow:
    bundle = await uow.bundles.lock_or_get(...)
    revision = await uow.revisions.create(...)
    await uow.members.create_many(...)
    await uow.jobs.enqueue(...)
    await uow.commit()
```

同一 Use Case 内所有 Repository 必须从同一个 `uow` 取得。并发写使用聚合根 `row_version` 条件更新或需要时的行锁；`commit` 冲突映射为稳定的领域错误，不泄漏 Oracle 异常。

## 外部资源与事务分段

数据库事务不得包裹长时间下载、病毒扫描、Parser 调用或模型调用。以文件受理为例：先独立提交 Receipt，再完成对象暂存；最终短事务只负责 KC 事实行和 Job。对象已发布而数据库提交失败时由 Receipt 补偿清理，不能为追求伪分布式事务而长时间持有 Oracle 连接。

Parse 回调、View 切换和 Bundle current Revision 切换各自是短 Application Service Use Case。回调必须在同一 UoW 内校验 `lease_owner/input_fingerprint`、写入产物、更新 Job 和目标状态；事务失败时回调整体失败，Worker 按协议重试或停止。

## 依赖注入与测试

FastAPI 在应用启动时创建 Engine、`kc_uow_factory`、ObjectStore、Parser URL/认证和服务审计器；Controller 通过依赖取得已构造的 Application Service，而非每次读取全局配置或新建 Client。请求认证生成不可变 `ActorContext`（actor、权限、Domain、request_id），显式传入 Command。

单元测试使用 Fake/In-memory Repository 或 Fake UoW 验证状态机和幂等；集成测试使用独立 Oracle Schema/事务验证约束、行锁和回滚；Adapter 测试独立模拟对象存储与 Parser 回调。任何 KC 测试都不得依赖 V1 File/Chunk 表。
