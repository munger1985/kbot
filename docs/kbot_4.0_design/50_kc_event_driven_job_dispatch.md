# KC 通知驱动任务调度

## 目标

Knowledge Core 的 Parser 与 Projection Worker 不再以固定两秒间隔持续空轮询。
`KBOT_KC_INGESTION_JOB` 仍是任务、重试、租约和恢复的唯一事实来源；
通知只负责提示 Worker 立即检查任务表，不能携带或替代任务。

## 事务与通知

`IngestionJobRepository.add()` 在任务成功 Flush 后登记唤醒通道。Unit of Work
在同一事务内调用 `DBMS_ALERT.SIGNAL`，Oracle 仅在事务提交后交付通知：

- `PARSE` → `KBOT_KC_PARSE_READY`
- `INDEX / PROFILE / COLLECTION_PURGE` → `KBOT_KC_PROJECTION_READY`

通知失败不会回滚持久任务。UoW 使用 Savepoint 隔离通知错误并继续提交，
Worker 由兜底扫描发现任务。

## Worker 行为

Worker 启动后先执行一次 Claim。队列为空时，通过独立 Oracle 连接执行
`DBMS_ALERT.WAITONE`，收到通知后立即继续 Claim；30 秒超时后进行一次兜底扫描。
通知连接不可用时自动退化为带随机抖动的指数退避：

```text
2s → 4s → 8s → 16s → 30s
```

成功领取任务后退避立即重置。重试任务的 `AVAILABLE_AT` 到期、Worker 重启、
通知合并或短暂断线均可由超时扫描恢复。

Projection Worker 使用统一
`POST /internal/v1/knowledge/projection-tasks/claim`，一次查询按
`COLLECTION_PURGE → INDEX → PROFILE` 优先级抢占任务。旧的三个独立 Claim
契约已删除；具体 Run、Heartbeat 和 Fail 契约仍按任务类型隔离。

## 并发与演进

多个 Worker 可能同时收到同一提示，最终仍由主键二次查询、
`FOR UPDATE SKIP LOCKED` 和有限租约确定唯一执行者。未来如果迁移 PostgreSQL 时，
保留 `JobWakeupPublisher/Listener` 端口并将适配器替换为 `LISTEN/NOTIFY`；
任务表和应用层状态机无需改变。

部署后可执行：

```bash
python tests/acceptance/check_kc_job_wakeup.py
```

验证服务账号能够注册 Alert，并在提交后收到通知。
