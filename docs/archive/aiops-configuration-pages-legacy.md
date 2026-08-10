# KBot 4.0 AIOps 配置页面字段与接口说明（历史归档）

> 本文记录旧前端页面与 `/api/v1/ops` 契约。本次只迁移后台 Python，且公开
> API 已统一到 `/api/v1/apps/aiops`，不得以本文作为当前后端实施依据。

本文是 AIOps 配置页面的前端产品与接口契约说明。它面向 Portal/APEX
页面设计与实现，不规定视觉排版；字段、枚举、状态和请求要求以 Main API
`/api/v1/ops` 的当前公开契约为准。

> **凭据设计变更中。** Target 的数据库用户名、密码不再要求用户先在服务器写入
> 环境变量，再填写 `env://` 路径。目标设计是在受控 API 中接收用户名和密码，使用
> 专用密钥加密后写入 AIOps 数据库，详情接口永不返回明文或密文。本节中的
> `diagnostic_credential` 与 `execution_credential` 是待实施的新公开契约；代码完成
> 前的线上接口仍使用 `diagnostic_secret_ref` 与 `execution_secret_ref`，不能混用。

## 1. 配置依赖与通用请求规则

配置资源按以下顺序创建和关联：

```text
Target ──┬── Agent Binding
         ├── Monitor Binding ← Monitor Source
         └── Inspection Plan Target ← Inspection Plan

Policy ─────→ Agent Binding
```

- 页面只调用 Main API 的 `/api/v1/ops/*`，不调用 `/internal/v1/*`，也不直接
  写入 `KBOT_OPS_*` 表。
- 创建资源和状态命令必须携带唯一 `Idempotency-Key` 请求头。同一次用户提交的
  重试必须复用该值。
- 修改资源、健康检查和状态命令必须携带详情 GET 或上一条命令响应提供的
  `If-Match: "rv-<row_version>"`。未获得 ETag 时禁用修改按钮；冲突后保留用户输入，
  重新读取详情后由用户决定是否再次提交。
- `schema_version` 由页面统一传递 `aiops.public.v1`，不作为可编辑字段。
- Target 数据库密码只在创建/更新请求中通过 HTTPS 发送一次；前端不记录、不回显、
  不写入 URL、浏览器日志、APEX Debug、页面状态或客户端存储。服务端只保存加密后的
  凭据。
- 监控 Token、Webhook Key 和 TLS 私钥暂仍使用 `SecretRef`；它们不属于本次 Target
  数据库凭据改造范围。

## 2. P55 Target 管理

Target 是受管 Oracle/MySQL 数据库。创建时 API 最小必填字段是
`display_name`、`db_type`、`environment`；要启用自动只读诊断，必须同时配置版本、
地址和只读诊断凭据。

创建接口：`POST /api/v1/ops/targets`

| 页面字段 | 请求参数 | 必填 | 含义与约束 | 示例 |
| --- | --- | --- | --- | --- |
| Target 名称 | `display_name` | 是 | 1–256 个字符的可读名称 | `ERP 生产主库` |
| 数据库类型 | `db_type` | 是 | `ORACLE` 或 `MYSQL` | `ORACLE` |
| 数据库版本 | `version_code` | 强烈建议 | 最长 64；用于选择版本化诊断目录，缺失时不能自动直连诊断 | `19c`、`8.0.36` |
| 环境 | `environment` | 是 | `PROD`、`STG`、`DEV` | `PROD` |
| 数据库角色 | `db_role` | 否 | `PRIMARY`、`STANDBY`、`UNKNOWN`；默认 `UNKNOWN` | `PRIMARY` |
| 主机 | `endpoint.host` | 配置地址时必填 | 主机名/IP，1–253 字符 | `oracle-prod.internal` |
| 端口 | `endpoint.port` | 配置地址时必填 | 1–65535 | `1521` |
| Oracle 服务名 | `endpoint.service` | Oracle 配置地址时必填 | Oracle 只能填写 `service`，不能填写 `database` | `ERPPRD` |
| MySQL 数据库名 | `endpoint.database` | MySQL 配置地址时必填 | MySQL 只能填写 `database`，不能填写 `service` | `erp` |
| TLS | `endpoint.tls_enabled` | 否 | 默认 `true` | `true` |
| 只读诊断用户名 | `diagnostic_credential.username` | 强烈建议 | 只读数据库账号；与密码成对提交 | `kbot_diag` |
| 只读诊断密码 | `diagnostic_credential.password` | 强烈建议 | 仅创建/轮换时输入，页面不回显 | `••••••••` |
| 执行用户名 | `execution_credential.username` | 否 | 经审批执行受控变更的独立账号；与密码成对提交 | `kbot_exec` |
| 执行密码 | `execution_credential.password` | 否 | 仅创建/轮换时输入，页面不回显 | `••••••••` |
| 安全等级 | `security_level` | 否 | 0–999，默认 1 | `3` |
| 能力声明 | `capabilities` | 否 | 值为 `true` 的键会被 Agent 视为已具备能力 | 见下文 |

Oracle 创建示例：

```json
{
  "schema_version": "aiops.public.v1",
  "display_name": "ERP 生产主库",
  "db_type": "ORACLE",
  "version_code": "19c",
  "environment": "PROD",
  "db_role": "PRIMARY",
  "endpoint": {
    "host": "oracle-prod.internal",
    "port": 1521,
    "service": "ERPPRD",
    "tls_enabled": true
  },
  "diagnostic_credential": {
    "username": "kbot_diag",
    "password": "仅在本次 HTTPS 请求中传输"
  },
  "execution_credential": {
    "username": "kbot_exec",
    "password": "仅在本次 HTTPS 请求中传输"
  },
  "security_level": 3,
  "capabilities": {
    "dynamic_performance_views": true,
    "dba_catalog_views": true,
    "replication_views": true,
    "session_management": true
  }
}
```

MySQL 将 `endpoint.service` 改为 `endpoint.database`，例如：

```json
{
  "host": "mysql-prod.internal",
  "port": 3306,
  "database": "erp",
  "tls_enabled": true
}
```

能力声明建议根据 `db_type` 展示固定复选项，而不是对普通用户暴露自由 JSON：

| 数据库 | 建议能力键 | 影响 |
| --- | --- | --- |
| Oracle | `dynamic_performance_views` | 会话、阻塞链、长事务诊断 |
| Oracle | `dba_catalog_views` | 存储容量诊断 |
| Oracle/MySQL | `replication_views` | 主备/复制状态诊断 |
| Oracle/MySQL | `session_management` | 会话类受控变更候选动作 |
| MySQL | `information_schema` | 会话、容量、长事务诊断 |
| MySQL | `sys_schema` | 阻塞链诊断 |

目标设计中，Target 表单直接提供“只读诊断账号”和“执行账号”两个凭据组。每组仅在
新建或“轮换凭据”时显示用户名、密码输入框；详情和编辑初始化仅返回
`configured`、`credential_id`、`key_version`、`updated_at` 等安全元数据，不返回用户名、
密码、密文、Nonce 或认证标签。更新普通 Target 字段时不携带凭据字段即表示保留原凭据；
显式“移除执行凭据”使用专门命令，不使用空密码作为歧义信号。

Target 创建后状态为 `MAINTENANCE`，健康状态为 `UNKNOWN`。详情页提供以下命令，
不提供物理删除：

| 用户动作 | 接口 | 允许的原状态 |
| --- | --- | --- |
| 启用 | `POST /api/v1/ops/targets/{target_id}/activate` | `MAINTENANCE` |
| 进入维护 | `POST /api/v1/ops/targets/{target_id}/maintenance` | `ACTIVE`、`DISABLED` |
| 停用 | `POST /api/v1/ops/targets/{target_id}/disable` | `ACTIVE`、`MAINTENANCE` |
| 编辑 | `PATCH /api/v1/ops/targets/{target_id}` | 任意状态，至少提交一个可修改字段 |

变更 `endpoint` 或轮换只读诊断凭据会使健康状态重置为 `UNKNOWN`；已启用 Target 会
自动回到 `MAINTENANCE`，应在保存前明确提示用户。

### 2.2 Target 数据库凭据的目标安全设计

此节记录待实施的改造边界，供前端、Main API、AIOps API、DB Executor 与数据库
脚本同步实现。

1. 前端仅在新建 Target、轮换只读诊断凭据或轮换执行凭据时提交
   `username`、`password`。密码控件不回填；取消编辑、普通保存或未修改凭据时均不
   发送密码。
2. Main API 只在服务端转发请求；日志、审计事件、幂等请求快照、错误响应和 APM
   属性均不得包含凭据字段。幂等记录需要对凭据字段进行不可逆摘要或排除保存，不能
   保存明文请求体。
3. AIOps API 生成 `credential_id`，使用专用于 AIOps 数据库凭据的 AES-256-GCM
   密钥加密用户名和密码。每条凭据生成独立随机 nonce，并使用
   `domain_id + credential_id + credential_kind` 作为附加认证数据（AAD）。数据库中
   不保存主密钥。
4. 凭据表保存 `credential_id`、`domain_id`、`credential_kind`、用户名密文、密码密文、
   nonce、`key_version`、审计列与软删除状态；Target 改为保存只读诊断和执行凭据的
   `credential_id` 外键。删除旧的 `*_SECRET_REF` 列、DTO 字段、Secret Store Target
   解析路径和兼容分支。
5. DB Executor 仍不持有 KBot Schema 数据库凭据，也不读取凭据表。诊断/变更 Grant
   仅携带对应 `credential_id`，不携带明文、密文或可解析路径。Executor 持 Grant 调用
   AIOps API 的受保护、短时、一次性内部凭据发放端点；AIOps API 校验服务身份、Grant
   audience、过期时间、Target/凭据匹配和一次性状态后才在内存中解密并返回账号密码。
   Executor 只在驱动调用期将其置于内存，随后清除引用；不得写日志、Artifact、事件或
   重试队列。
6. 主密钥由部署 Secret 注入，不经 Portal 配置。新增用途隔离的
   `KBOT_AIOPS_CREDENTIAL_ENCRYPTION_KEY`，不复用现有认证加密密钥；密钥格式必须
   严格为 32 字节随机值。支持 `key_version` 与当前/历史密钥环，以便先轮换密钥、再
   后台重加密存量凭据，最后移除旧密钥。

建议新增的 Target 请求片段：

```json
{
  "diagnostic_credential": {
    "username": "kbot_diag",
    "password": "仅本次请求提供"
  }
}
```

Target 详情中的安全返回片段：

```json
{
  "diagnostic_credential": {
    "configured": true,
    "credential_id": "019f8eae-2c25-7d48-b044-350ec3f5a111",
    "key_version": "2026-08",
    "updated_at": "2026-08-03T08:00:00Z"
  }
}
```

凭据轮换采用独立命令而不是常规 PATCH：

- `POST /api/v1/ops/targets/{target_id}/diagnostic-credential:rotate`
- `POST /api/v1/ops/targets/{target_id}/execution-credential:rotate`
- `POST /api/v1/ops/targets/{target_id}/execution-credential:remove`

上述命令均需 `If-Match` 和 `Idempotency-Key`。只读诊断凭据轮换会将 Target 置为
`MAINTENANCE` 并重置健康状态；移除执行凭据只禁止自动变更执行，不影响只读诊断。

已有环境不保留旧 `env://` Target 凭据兼容路径。先执行
`scripts/db/aiops_target_credentials_expand.sql` 创建凭据表和新外键列；部署新服务后，
管理员在 Target 页面重新录入凭据。确认所有 Target 已重新配置后，执行
`scripts/db/aiops_target_credentials_cleanup.sql` 删除旧 `*_SECRET_REF` 列。清理脚本在
检测到仍有旧引用时会直接拒绝执行。

### 2.1 Target 的 Agent Binding 页签

创建接口：`POST /api/v1/ops/targets/{target_id}/agent-bindings`

| 页面字段 | 请求参数 | 必填 | 含义与约束 | 示例 |
| --- | --- | --- | --- | --- |
| AIOps Agent | `agent_id` | 是 | 从 `GET /api/v1/agents` 选择；候选必须属于当前 Domain 且具备 `aiops` 能力 | UUID |
| 允许执行变更 | `allow_mutation` | 否 | 默认 `false`；关闭时只允许诊断与建议 | `false` |
| 执行策略 | `policy_id` | 否 | 选择 Policy；实际执行需要有效且允许执行的 Active Policy | UUID |
| 允许动作 | `allowed_actions` | 否 | 允许动作技术名数组 | `["session.kill"]` |
| 变更窗口 | `change_window` | 否 | 结构化 JSON；当前没有固定公开子字段 | 由后续规则定义 |
| 每日执行上限 | `max_daily_executions` | 否 | 非负整数 | `5` |

页面应先显示“只诊断”默认模式；勾选“允许执行变更”后才展示策略、动作、变更窗口和
上限。一个 Agent 不可重复绑定到同一 Target。创建或恢复绑定时，Main API 会同步
Agent 的 `config.aiops_target_id`；撤销当前绑定时会清除它。前端不得维护或提交
`config.aiops_agent_id`。

绑定编辑使用 `PATCH /api/v1/ops/targets/{target_id}/agent-bindings/{binding_id}`；
撤销/恢复使用 `POST .../{binding_id}/revoke` 或 `POST .../{binding_id}/restore`。

## 3. P56 监控源管理

Monitor Source 表示可复用的 Prometheus、Zabbix 或 OEM 实例；它本身尚未指定对应
哪一个 Target。

创建接口：`POST /api/v1/ops/monitor-sources`

| 页面字段 | 请求参数 | 必填 | 含义与约束 | 示例 |
| --- | --- | --- | --- | --- |
| 监控源名称 | `display_name` | 是 | 1–256 个字符 | `生产 Prometheus` |
| 监控类型 | `source_type` | 是 | `PROMETHEUS`、`ZABBIX`、`OEM` | `PROMETHEUS` |
| 服务地址 | `endpoint` | 是 | HTTP/HTTPS URL；不得带用户名、密码、query 或 fragment | `https://prometheus.example.com` |
| 查询凭据引用 | `secret_ref` | 否 | 访问监控系统所需凭据 | `env://KBOT_AIOPS_PROM_TOKEN` |
| Webhook 密钥引用 | `webhook_secret_ref` | 否 | 验证告警回调签名 | `env://KBOT_AIOPS_PROM_WEBHOOK` |
| TLS 配置引用 | `tls_profile_ref` | 否 | 私有 CA 或 mTLS 配置引用 | `vault://kbot/tls/prometheus` |
| 扩展能力 | `capabilities` | 否 | 对接适配器的扩展声明 | `{"external_target_label":"instance"}` |

`capabilities.external_target_label` 如提供，必须是合法监控标签名：以字母或下划线
开始，只包含字母、数字、下划线，最长 128 字符。

```json
{
  "schema_version": "aiops.public.v1",
  "display_name": "生产 Prometheus",
  "source_type": "PROMETHEUS",
  "endpoint": "https://prometheus.example.com",
  "secret_ref": "env://KBOT_AIOPS_PROM_TOKEN",
  "webhook_secret_ref": "env://KBOT_AIOPS_PROM_WEBHOOK",
  "capabilities": {"external_target_label": "instance"}
}
```

创建后状态为 `DISABLED`。详情页提供：

- 编辑：`PATCH /api/v1/ops/monitor-sources/{source_id}`；
- 健康检查：`POST /api/v1/ops/monitor-sources/{source_id}/health-checks`，返回已受理回执，
  页面轮询详情中的 `health_check_pending`、`health_status`、`last_error_code`；
- 启用/停用：`POST .../{source_id}/enable`、`POST .../{source_id}/disable`；
- 轮换 Webhook Key：`POST .../{source_id}/webhook-key:rotate`。新 Key 只显示一次，
  不写入页面项、调试日志或报表。

### 3.1 Target 的 Monitor Binding 页签

该页签将一个 Monitor Source 中的外部对象映射到当前 Target。

创建接口：`POST /api/v1/ops/targets/{target_id}/monitor-bindings`

| 页面字段 | 请求参数 | 必填 | 含义与约束 | 示例 |
| --- | --- | --- | --- | --- |
| 监控源 | `source_id` | 是 | 选择已创建 Monitor Source | UUID |
| 外部对象标识 | `external_target_key` | 是 | 在监控系统中定位此数据库的键，1–256 字符 | `oracle-prod-01:1521/ERPPRD` |
| 绑定角色 | `role` | 否 | `PRIMARY` 或 `SUPPLEMENTARY`，默认 `PRIMARY` | `PRIMARY` |
| 优先级 | `priority` | 否 | 非负整数，默认 100；值越小优先级越高 | `10` |
| 指标范围 | `metric_scope` | 否 | 限定读取的资源/指标，当前无固定子字段 | `{"service":"erp"}` |
| 映射覆盖 | `mapping_overrides` | 否 | 覆盖适配器默认字段映射，当前无固定子字段 | `{"instance_label":"db_instance"}` |

```json
{
  "source_id": "019f8eae-2c25-7d48-b044-350ec3f5a111",
  "external_target_key": "oracle-prod-01:1521/ERPPRD",
  "role": "PRIMARY",
  "priority": 10,
  "metric_scope": {"service": "erp"}
}
```

`metric_scope` 与 `mapping_overrides` 尚无公开强类型 schema。普通用户页面不应默认
显示自由 JSON 编辑器；在具体 Provider 的映射规范确定后转为控件。高级管理员可在
明确告知风险后使用受校验 JSON 编辑区。

列表：`GET /api/v1/ops/targets/{target_id}/monitor-bindings`；编辑：
`PATCH .../{binding_id}`；启用/停用：`POST .../{binding_id}/enable` 或
`POST .../{binding_id}/disable`。

## 4. P57 Policy 管理

Policy 定义 Agent 可观察、诊断、建议和执行的边界。Policy 是不可变版本资源：修改
规则时使用相同 `policy_key` 创建一个新版本，而不是 PATCH 原版本；激活新版本时，同
`policy_key` 的旧 Active 版本会自动退役。

创建接口：`POST /api/v1/ops/policies`

| 页面字段 | 请求参数 | 必填 | 含义与约束 | 示例 |
| --- | --- | --- | --- | --- |
| 稳定策略标识 | `policy_key` | 是 | 小写字母开头；可含数字、`.`、`_`、`-`；最长 128 | `prod-db-safe` |
| 策略名称 | `display_name` | 是 | 1–256 个字符 | `生产数据库安全策略` |
| 策略规则 | `rules` | 是 | `ops.policy.v1` 规则对象 | 见下表 |

`rules` 的当前正式字段如下：

| 页面字段 | 请求参数 | 必填 | 含义与约束 | 示例 |
| --- | --- | --- | --- | --- |
| 规则版本 | `rules.schema_version` | 是 | 固定为 `ops.policy.v1` | `ops.policy.v1` |
| 允许 Agent 执行 | `rules.allow_agent_execution` | 是 | 是否允许实际变更 | `false` |
| 最大风险 | `rules.max_risk_level` | 否 | 默认 `LOW` | `LOW`、`MEDIUM`、`HIGH`、`CRITICAL` |
| 允许动作 | `rules.allowed_action_types` | 否 | 非空字符串数组，默认 `[]` | `["session.kill"]` |
| 自动观察最低级别 | `rules.auto_observe_min_severity` | 否 | 默认 `CRITICAL` | `INFO`、`WARNING`、`HIGH`、`CRITICAL` |
| 告警冷却秒数 | `rules.alert_cooldown_seconds` | 否 | 0–86400，默认 900 | `900` |

当前运行时还会读取以下高级键；它们尚未形成完整的公开强类型 schema，应标注为高级
配置，并在产品确认规则后再做字段化控件：

| 高级键 | 含义 | 示例 |
| --- | --- | --- |
| `rules.readonly_database_enabled` | 是否允许只读数据库直连诊断 | `true` |
| `rules.aiops_collection_ids` | AIOps 可检索的知识库 Collection UUID 数组 | `["019f8eae-..."]` |
| `rules.entitlements` | 诊断/动作目录要求的授权标识数组 | `["enterprise"]` |

```json
{
  "schema_version": "aiops.public.v1",
  "policy_key": "prod-db-safe",
  "display_name": "生产数据库安全策略",
  "rules": {
    "schema_version": "ops.policy.v1",
    "allow_agent_execution": false,
    "readonly_database_enabled": true,
    "max_risk_level": "LOW",
    "allowed_action_types": [],
    "auto_observe_min_severity": "CRITICAL",
    "alert_cooldown_seconds": 900,
    "aiops_collection_ids": ["019f8eae-2c25-7d48-b044-350ec3f5a111"]
  }
}
```

创建后状态为 `DRAFT`。状态命令为：

- 激活：`POST /api/v1/ops/policies/{policy_id}/activate`；
- 退役：`POST /api/v1/ops/policies/{policy_id}/retire`。

Policy 没有 PATCH 接口。页面应以“基于当前版本创建新草稿”的交互替代编辑原版本，
并展示 `version_no`、`policy_hash`、`effective_at` 和 `retired_at`。

## 5. P58 巡检计划与范围

巡检由计划定义调度和模板，再单独添加 Target 范围。创建计划后默认为 `PAUSED`；
至少有一个 Active Target 才能激活。

### 5.1 创建巡检计划

创建接口：`POST /api/v1/ops/inspection-plans`

| 页面字段 | 请求参数 | 必填 | 含义与约束 | 示例 |
| --- | --- | --- | --- | --- |
| 计划名称 | `display_name` | 是 | 1–256 个字符 | `生产库每日健康巡检` |
| 调度类型 | `schedule_type` | 是 | `DAILY`、`WEEKLY`、`CRON` | `DAILY` |
| Cron 表达式 | `cron_expression` | 是 | 规范五段 Cron，9–256 字符 | `0 2 * * *` |
| 时区 | `timezone` | 是 | IANA 时区 | `Asia/Shanghai` |
| 巡检模板 | `template_id` | 是 | 已在部署中登记的模板 ID | `database-health` |
| 模板版本 | `template_version` | 是 | 已登记版本 | `1.0.0` |
| 超时秒数 | `timeout_seconds` | 是 | 1–86400 | `1800` |
| 重叠策略 | `overlap_policy` | 否 | 默认 `SKIP` | `SKIP`、`QUEUE` |
| 补跑策略 | `misfire_policy` | 否 | 默认 `LATEST_ONLY` | `SKIP`、`LATEST_ONLY` |
| 调度解析器版本 | `schedule_resolver_version` | 是 | 必须匹配部署模板登记值 | `v1` |

```json
{
  "schema_version": "aiops.public.v1",
  "display_name": "生产库每日健康巡检",
  "schedule_type": "DAILY",
  "cron_expression": "0 2 * * *",
  "timezone": "Asia/Shanghai",
  "template_id": "database-health",
  "template_version": "1.0.0",
  "timeout_seconds": 1800,
  "overlap_policy": "SKIP",
  "misfire_policy": "LATEST_ONLY",
  "schedule_resolver_version": "v1"
}
```

后端无论 `schedule_type` 取何值都会要求 `cron_expression`。因此：

- `DAILY` 页面选择时间后生成 Cron，例如凌晨 02:00 生成 `0 2 * * *`；
- `WEEKLY` 页面选择星期与时间后生成 Cron，例如周一 02:00 生成 `0 2 * * 1`；
- `CRON` 才显示五段表达式编辑控件，并即时校验 IANA 时区和未来 370 天内是否存在下次触发时间。

`template_id`、`template_version`、`schedule_resolver_version` 必须匹配部署时登记的
巡检模板。当前 Main API 没有公开的模板目录查询接口，页面不能自行猜测或硬编码
模板和 `template_overrides` 字段。实现前须由部署配置注入受允许模板，或补充只读模板
目录 API。

### 5.2 管理巡检 Target 范围

添加接口：`POST /api/v1/ops/inspection-plans/{plan_id}/targets`

| 页面字段 | 请求参数 | 必填 | 含义与约束 | 示例 |
| --- | --- | --- | --- | --- |
| 巡检 Target | `target_id` | 是 | 从当前 Domain 的 Target 列表选择；同一计划不可重复添加 | UUID |
| 模板覆盖参数 | `template_overrides` | 否 | 键必须属于模板登记的允许覆盖键 | `{"threshold_percent":85}` |

添加或编辑计划 Target 时使用**计划**的 ETag，而不是 Target 自己的版本。已激活的
计划必须始终保留至少一个 Active Target。

计划状态命令：

| 用户动作 | 接口 | 前置条件 |
| --- | --- | --- |
| 激活 | `POST /api/v1/ops/inspection-plans/{plan_id}/activate` | 至少一个 Active Target；模板、Cron、时区有效；范围不超过部署上限 |
| 暂停 | `POST /api/v1/ops/inspection-plans/{plan_id}/pause` | 当前为 `ACTIVE` |
| 停用 | `POST /api/v1/ops/inspection-plans/{plan_id}/disable` | 当前为 `ACTIVE` 或 `PAUSED` |
| 编辑计划 | `PATCH /api/v1/ops/inspection-plans/{plan_id}` | 至少提交一个可修改字段 |

## 6. 页面验收清单

- 所有选择列表仅展示当前 Domain 内的资源；ID 是值，名称和状态是辅助展示。
- 任何凭据字段都只显示“已配置/未配置”和指纹，绝不回显密钥引用的实际内容或密钥值。
- Target、Monitor Source、Binding、Policy、Inspection Plan 均无物理删除入口。
- 所有状态动作均使用固定按钮和确认文案，命令名不是用户可编辑输入。
- 列表使用 API 的 cursor 分页参数：`status`、`cursor`、`limit`；`limit` 范围为 1–200，默认 50。
- API 返回 `412`、`428`、`409`、`422` 或 `503` 时，页面显示可行动的安全提示和 `request_id`，不显示堆栈、数据库错误或 Secret。

## 7. 契约来源

- `packages/platform_core/src/platform_core/contracts/aiops/configuration.py`
- `services/main_api/src/main_api/api/ops.py`
- `services/aiops_agent/src/aiops_agent/application/configuration/`
- `docs/openapi/aiops_public_v1.json`
