# AIOps Agent 目录故障隔离与 Ammolite 定向修复

## 线上原因与授权范围（2026-11-11）

Ammolite 21:17 在线 development-logs 记录：运维知识库列表因 `aiops_knowledge_base.configuration_json` 缺失返回 500，Main API 对应 `OPS_UPSTREAM_UNAVAILABLE`。实际同一 Domain 有 1 个 `Cyber DBA` Agent、4 个已上线业务系统；前端将 Agent、系统、模型、标签和知识库放在同一组 `Promise.all` 后统一赋值，一个失败会让全部初次加载结果丢失。共用错误变量还会被向导校验清空。

用户批准：修复并部署各目录独立加载/错误提示，定向补齐此次报错的规范字段，保留既有数据；经正常应用会话建立 Ammolite Exadata 模拟系统、硬件及 OEM 绑定。未授权全库初始化、删除旧解析、修复其余知识结构或扩大现有 Agent 授权。

## 两端一致的页面行为

- Agent、系统、模型、标签、知识库分别处理结果和状态。成功目录即时可用，失败目录明确标名；刷新失败保留上次成功数据并标为仅供参考，不把失败当作空目录。
- 目录错误与表单校验错误分开，在页面和抽屉持续显示；切换步骤不清除目录错误。抽屉可重试目录，名称、模型和授权草稿不因重试被重置。
- 业务系统目录失败、格式无效或尚有未读取的分页时，不把授权 ID 当成已删除对象进行过滤；草稿保留原 ID 并禁止提交范围。选择器明确显示目录未就绪/失败，不显示假的“没有匹配项”。只有成功完整读取的目录才可按既有规则移除真正不存在的系统。
- 知识库/标签读取失败不修改原知识授权；未启用知识问答的诊断 Agent 仍可保留原配置保存。没有在前端自动关闭能力、清空知识策略、追加权限或选择特定厂商模型。
- Ammolite Vue 和 KBot 静态 Agent 页面同步实施；普通监控、诊断、聊天和后端 HTTP 合同不变。

## PostgreSQL 定向部署

规范 Entity 的 `configuration_json` 为非空 JSONB，应用默认空字典。本次不新增持久化合同，仅让显式初始化器具备已有规范字段的定向对齐入口：

```bash
python scripts/deployment/init_postgresql.py --align-aiops-knowledge-base-configuration-only
```

生产/开发连接均通过原部署配置或现行服务 Settings 解析，不向命令行或日志打印 DSN、口令。线上可在受控诊断进程中复用同一个 `_align_aiops_knowledge_base_configuration`，只针对已确认的 `ammolite` schema 执行单事务变更。

- 缺列时仅新增 `configuration_json JSONB NOT NULL`，用临时 `DEFAULT '{}'::jsonb` 保留旧知识库为“尚未配置”，随后移除数据库默认值以保持规范。没有猜测模型、表示空间或解析 Profile。
- 已有规范列时无写入；已有列类型/可空性不符时明确拒绝自动改写。Schema 标识符严格校验，锁等待限 5 秒，失败则 PostgreSQL 事务回滚。
- 不调用 `initialize`、`reset_aiops_schema`、`create_all` 或其他升级分支；不更新/删除业务行，不创建另外 19 张缺失知识表，也不处理旧索引表的 5 个缺失字段。
- 正常全量初始化也复用该字段对齐方法；此检查只在显式部署工具内，绝不放入服务启动、readiness、UoW 或业务请求。
- 部署前后复核旧字段内容/行数、Agent/系统/绑定与配置摘要、目标列及其他表列元数据。此变更和前端发布不需要重启任何服务。

## 验证与交付状态

本地限定 Python 回归 68 项通过（Agent 配置 API、知识库、6 项定向结构合同及初始化器），Ammolite/KBot 架构边界检查和 Ammolite 106 张 AIOps ORM metadata 检查通过；Portal typecheck、生产构建（957 modules，仍有既有大 chunk 告警）及 KBot 脚本语法检查通过。

两端真实组件/静态脚本的浏览器夹具共 43 项：42 项功能通过，其中本次新增 12 项目录失败、授权保留、跨步骤提示和重试恢复测试。一项既有视觉对比仍失败：KBot 抽屉阴影透明度 0.20、Ammolite 0.16。已用修改前的干净 tree `/tmp/exadata-chat-kbot-FFOPtx` 与 `/tmp/exadata-chat-ammolite-DVGshV` 复现同一断言，未放宽断言或改动无关样式。这是受控夹具回归，不是线上正常登录后的页面验收。

### 实际部署记录（2026-11-11 21:29，UTC+8）

两端代码已推送并在 kbotdev 精确 fast-forward：Ammolite `b8e9d515`、KBot `69fff706`。部署前确认原 HEAD、已跟踪工作区和暂存区，核对提交的完整文件清单；原有未跟踪配置备份保留。旧文件、配置摘要、数据库前后核验和进程基线保存在主机私有目录 `/home/ubuntu/aiops-agent-catalog-20261111-qicd_ty_`。

- Ammolite 单事务新增 `aiops_knowledge_base.configuration_json`，核验为 `JSONB NOT NULL` 且无数据库默认值；原有 1 个知识库配置为 `{}`。所有 `aiops_*` 表列元数据前后只有这一列新增，第二次调用定向 helper 无写入。未处理另外 19 张缺失知识表或旧索引字段。
- 16 张受保护表的旧字段内容指纹及行数前后相同，包括 1 个 Agent、4 个 Agent 版本、12 条版本/系统授权、4 个业务系统、6 个资源及 11 条监控绑定；未改变已有 Agent 授权或业务数据。
- 在现行服务配置的受控子进程中，通过真实 UoW/Repository 重新读取知识库列表及 usage，均成功返回 1 项。此项是内部仓储读验证，不是正常会话公共 API 验收。
- Ammolite 在线 Vue 模块 HTTP 200，包含新目录状态及重试逻辑；KBot `/ui/aiops/agents.html` 与 `/ui/aiops/js/aiops-agents.js` 均 HTTP 200，响应 Hash 与已部署文件一致。
- 52 个基线进程 PID/启动时间全部不变，两端 `.env` 和 TOML 配置 Hash 不变。两端 Main、AIOps API、Worker 共 6 个实际 readiness 均 HTTP 200；本轮没有重启任何服务。

此时尚未取得正常 Ammolite 应用会话，未宣称完成登录后列表/选择器的线上浏览器验收，也未创建模拟系统。用户随后提供测试账号，经正常登录完成的实际验收见下节。

## Exadata 模拟系统

现有 Ammolite `OEM24ai` 已 ENABLED/CONNECTED，沿用标准托管凭据及真实 OEM 中的 `AIOPS_SIM_EXADATA_DBM01/GRID01/CELL01`，无需新模拟容器。必须通过正常应用会话的公共发现/业务系统 API 创建独立模拟系统、EXADATA 硬件和来源绑定；不复制 KBot 主键、绑定行或加密凭据，不修改现有 Agent 授权。若没有正常会话，明确报告该部分未完成，不伪造 SSO、重置密码、直接写业务表或把“代码已部署”说成“模拟系统已配置”。

### 正常登录后的实际验收（2026-11-11 21:43—21:45，UTC+8）

用户提供测试账号后，通过标准 SSO 密码登录、Bootstrap 和指定 Tenant/App/Domain 的 BUSINESS 会话进入 AIOps 默认空间。没有签发伪造 Token、借用 KBot 会话、重置密码或直接写业务表；凭据不写入仓库、文档或诊断输出。

- 正常公共 API 确认原有 1 个 `Cyber DBA`、4 个已上线业务系统、1 个可读取知识库。通过现有 `OEM24ai` 发现唯一 SIM DB Machine 及其 Grid/Cell，以服务端候选引用原子创建并上线 `Exadata OEM 模拟联调（SIM）`，硬件为 `SIM - Exadata DBM01`；使用现有 `dba team`，环境为 DEV，描述/属性明确为合成联调数据。
- 系统 ID `01a1cad9-2904-7892-8428-fc333bd1f7ec`、EXADATA 资源 ID `01a1cad9-2906-7c00-baa6-1306ba23b359`、绑定 ID `01a1cad9-2940-783f-b096-b200570920d7`。均为 Ammolite 新身份，不复制 KBot 主键；监控目录为 READY，只读诊断就绪度仍诚实显示 PARTIAL。
- Chromium 经 SSH 隧道加载已部署 Portal，使用正常 Cookie 会话，无静态资源替换、无 API 路由 mock。Agent 管理页实际显示 `Cyber DBA`；新建向导中 5 个业务系统均可选，成功选中原 OEM 系统和新模拟系统后取消，未保存 Agent 或变更授权。
- 实时监控页面实际显示整机 5 张快照卡、Cell 12 张快照卡；Cell 历史 12 张卡、10 张数值图、570 条真实 OEM 返回的模拟采样。CPU `48.3%`、内存 `64.43%`、温度原值 `32.0`（OEM 原单位为空），采样时间为北京时间 `21:42:00`；DOM 原值和采集时间与响应一致。1600 宽度三列且无横向溢出，无 pageerror 或失败响应。
- 通过公共 API 复核原有 4 个系统的完整详情及 Agent 列表/授权均未改变；OEM 来源完整响应前后一致，未修改任何来源配置。没有为聊天自动授权新系统，没有新增模拟容器、执行 DDL、同步 Prompt 或重启服务。

远端私有验收目录为 `/home/ubuntu/ammolite-exadata-sim-20261111-5e8119hr`，本地浏览器截图/响应/结果位于 `/tmp/ammolite-exadata-verify-4Ncin4`；会话仅存私有目录，不提交凭据或会话。指标仍是 OEM 中的合成值，不代表物理设备实测、秒级采集或健康阈值已验证。本轮未生成新的诊断报告，也未验收 Ammolite 聊天。
