# KBot 4.0 开发日志页面

`tools/dev_console/` 只保留 development 环境的运行日志与 API 访问日志查看页面，
不再提供 Knowledge Core、Agent、AIOps 或 Run 调试测试页面。正式 KM 页面位于
`ui/km/`。

启动静态服务器：

```bash
python3 tools/dev_console/server.py --port 8080
```

访问 `http://127.0.0.1:8080/` 或
`http://127.0.0.1:8080/operations-logs.html`。根路径会直接跳转到日志页面。

日志页面根据当前浏览器主机自动连接 Main API 的 `18099` 端口，不在页面配置
Main API URL、Domain ID 或 User ID。页面只调用 development 环境开放的
`/api/v1/development/logs/*`。这些只读端点匿名开放，页面不保存、发送或要求用户
Token、App API Key 和测试身份 Header。

执行 `./start_kbot.sh` 时，静态服务在 `ENVIRONMENT=development` 下默认随 KBot
启动，并由 `./stop_kbot.sh` 一并停止。可通过
`KBOT_UI_ENABLED=false ./start_kbot.sh` 临时关闭；非 development 环境即使显式
开启该变量也不会启动。

匿名日志接口只允许在 development 环境启用，生产环境不会注册这些路由。
