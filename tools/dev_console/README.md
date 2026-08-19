# KBot 4.0 UI 测试页

这些页面是面向本地开发环境的 Vanilla JavaScript 测试工具，不模拟 KM Portal，
也不应部署到公网。它们只调用 Main API 的公开 `/api/v1`，不会访问任何内部接口。

启动静态服务器：

```bash
python3 tools/dev_console/server.py --port 8080
```

执行 `./start_kbot.sh` 时，UI 在 `ENVIRONMENT=development` 下默认随 KBot
启动，并由 `./stop_kbot.sh` 一并停止。可通过
`KBOT_UI_ENABLED=false ./start_kbot.sh` 临时关闭；非 development 环境即使显式
开启该变量也不会启动 UI。

访问 `http://127.0.0.1:8080`。开发配置必须设置
`development_auth_bypass = true`；开发环境会自动允许本机 UI Origin。页面发送
`X-KBot-Test-Auth: true`，因此不需要用户 Token或 App API Key；仍需填写数据库中有效的
Domain ID 和测试 User ID。

- `knowledge-core.html`：按资源准备、文件入库、Agent 配置和诊断输出四个阶段组织。
  支持创建 Domain、Collection 和 Agent；一个文件可作为单文件 Bundle 入库，多个
  文件可分别入库或共同组成 Bundle。上传记录会自动跟踪审批、解析、索引和最终
  可检索状态；创建或点选 Agent、Collection 后可直接完成绑定。页面不读取 MetaDB，
  也不构造 Asset。
- `agent-chat.html`：选择 Agent、创建 Conversation、提交文字
  或图片问题、读取带认证 Header 的 SSE、查看 Trace、历史和最终 Artifact。
- `aiops.html`：AIOps 配置页。创建独立 AIOps Agent、数据库/监控资源及诊断策略，
  完成 KC、Target-Agent 与 Monitor-Target 绑定，并管理巡检计划和报告。
- `aiops-chat.html`：独立的 AIOps 对话诊断台。创建或恢复 Chat Run，通过 SSE
  展示监控采集、数据库取证、知识检索和 LLM 推理；数据库不可直连时展示受控只读
  SQL，允许逐项粘贴结果后继续同一个 Run。变更命令必须逐条审批，也支持
  自动变更能力不可用时回填人工执行结果，随后继续验证、对比和报告生成。
- `agent-debug.html`：按 Run 聚合 Task、Event、Artifact、KC 两阶段检索诊断和
  跨服务运行日志，用于重放一次聊天的完整执行过程。
- `operations-logs.html`：不关联具体 Run 的通用日志文件浏览器。

若页面通过 `file://` 直接打开，浏览器可能因 Origin/CORS 策略拒绝请求，因此应使用
上面的 HTTP Server。

测试绕过只允许在 `development` 环境启用，生产配置必须保持关闭。它只绕过公开
Main API 的公开身份校验；Domain 校验和所有 `/internal/v1` 认证仍然生效。
