# AIOps 网络边界

AIOps 控制面与目标库网必须分离。本期把它做成可验收部署约束，不靠口头说明。

## 1. 必须遵守的边界

- Portal、Main API、模型服务配置禁止出现目标数据库地址、账号或口令。
- 只有库网内的 DB Executor / Collector 持有诊断凭据和执行凭据。
- AIOps API 与 DB Executor 必须使用不同服务身份，禁止共用 `service_name`。
- 跨服务依赖 URL 只能是 `http`/`https` 主机地址，禁止携带用户名、口令、查询串或片段。
- Target 对外详情只返回主机、端口、服务名/库名和凭据是否已配置；不返回口令、完整 DSN 或连接串。
- 前端不得展示、缓存或回填 DSN / 口令。写入凭据只走一次性提交，随后只显示“已配置”。
- 不能直连目标库时，只使用监控、日志和用户补证；Finding / Gap 必须写明当前没有库内取证，而不是伪装成正常。

## 2. 部署核对清单

1. `kbot-aiops-api` 与 `kbot-aiops-db-executor` 的 `service_name` 不同。
2. API / Worker / Scheduler 配置中的 `model_serving`、`knowledge_core`、`aiops_api`、`db_executor` 地址不含账号口令。
3. 目标库主机、端口、服务名只出现在 Target Endpoint 和 Executor 运行环境。
4. 诊断凭据、执行凭据只注入 Executor 侧 Secret；Portal 和 Main API 进程环境看不到这些口令。
5. 浏览器网络面板中的 Target 详情响应不含 `password`、DSN 或完整连接串。
6. 前端 Target 表单在保存成功后清空口令输入框，后续只显示凭据状态。

## 3. 负向验收

以下配置必须启动失败或被契约拒绝：

- API 与 Executor 使用相同 `service_name`。
- 任一 AIOps 依赖 `base_url` 写成 `http://user:password@host:port`。
- 把数据库口令写入 Portal / Main API / 模型服务配置。
- 通过 Target 详情接口回读口令。

不能直连时的产品行为：

- 自动告警/巡检仍可基于监控和日志出 Finding 或 Gap。
- 不得把“未取得库内会话行”写成“当前没有阻塞”。零行只在对应 Tool 已经跑过时成立。

## 4. 不在本期改动的部分

现有控制面与 Executor 分离模型保持不变。本期只补部署约束、契约检查和负向测试，不把凭据下沉路径改到 API 进程。
