# 开发日志浏览台

## 定位

该页面用于 KBot 4.0 开发和联调阶段读取本机日志，不属于 Portal 产品页面，也不
写入数据库。Main API 仅在 `platform.debug=true` 时注册
`/api/v1/development/logs/*`。

## 日志目录

日志按逻辑服务聚合，而不是按独立进程拆分：

```text
var/log/
├── main_api/{runtime.log,access.log}
├── knowledge_core/{runtime.log,access.log}
├── agent_runtime/{runtime.log,access.log}
├── model_serving/{runtime.log,access.log}
└── aiops_agent/{runtime.log,access.log}
```

`runtime.log` 保存业务流程、Worker、模型调用、启动/停止和异常堆栈；
`access.log` 每个 HTTP 请求只保存一条完成态记录。多个进程通过日志中的
`process` 区分，例如 `knowledge_core/api`、`knowledge_core/parser` 和
`knowledge_core/projection`。

启动脚本不再创建 `startup/`。解释器导入错误、端口冲突、退出码和 Unix Signal
统一追加到所属服务的 `runtime.log`。Access 事件不会重复写入 Runtime。

## 写入与轮转

Loguru 使用目录级文件锁协调同一逻辑服务的多个进程。每个服务只有
`runtime.log` 和 `access.log` 两个活动文件；轮转文件使用
`runtime.log.<timestamp>`、`access.log.<timestamp>` 命名，并按平台保留周期
清理。直接启动进程时保留控制台输出，`start_kbot.sh` 托管时关闭控制台副本，
避免相同日志重复落盘。

日志至少包含时间、级别、进程、代码位置和消息，并尽可能保留 `error_id`、
`request_id`、`trace_id`、HTTP 状态码和耗时。4xx Access 使用 WARNING，5xx
使用 ERROR；健康探针和空任务领取仅在 DEBUG 级别记录。

## 查询模型

`main_api.developer_tools.LogFileCatalog` 只扫描：

```text
var/log/<service>/runtime.log*
var/log/<service>/access.log*
```

它拒绝符号链接，并确保路径位于配置日志根目录内。每个文件最多读取最近 2 MiB，
再按时间倒序合并。

开发 API：

- `GET /api/v1/development/logs/services`：返回服务及两类日志摘要；
- `GET /api/v1/development/logs/events`：必须指定 `service_name` 和
  `log_type=RUNTIME|ACCESS`，可继续按级别、关键字和条数过滤。

页面只提供服务选择、运行/访问日志切换、级别、关键字和刷新频率。点击日志行可
查看完整多行原文。
