# AIOps 数据库凭据加密改造收口计划（历史归档）

> 专用凭据表和专用密钥方案已由平台统一 `KBOT_MANAGED_CREDENTIAL`
> 与 `KBOT_MANAGED_CREDENTIAL_KEY` 取代。当前基线见
> `../migrations/ammolite-aiops-knowledge-backend.md`。

## 1. 当前状态

本次改造目标是：Target 页面直接接收数据库用户名和密码，服务端使用专用 AES-256-GCM
密钥加密后写入 `KBOT_OPS_CREDENTIAL`；Target 只保存凭据 ID；详情接口不返回账号、密码、
密文或 SecretRef；DB Executor 仅在短时授权窗口内取得凭据。

当前已完成或已存在：

- 已创建数据库扩展脚本和清理脚本：
  - `scripts/db/aiops_target_credentials_expand.sql`
  - `scripts/db/aiops_target_credentials_cleanup.sql`
- 已在 `.env` / `.env.example` 增加：
  - `KBOT_AIOPS_CREDENTIAL_ENCRYPTION_KEY`
  - `KBOT_AIOPS_CREDENTIAL_KEY_VERSION`
- 已移除 `AIOPS_DEMO_ORACLE_READONLY` 示例变量。
- 已部分切换 Target DTO、Target ORM、运行快照和 Grant 字段。
- 已新增 `CredentialEntity`，但尚未完成 Repository、加密写入和运行时取用。

当前不能启动 AIOps：Target 配置服务、Executor 和部分测试仍未完成统一切换。

## 2. 实施原则

- 不保留旧 Target `env://`、`SecretRef`、`*_SECRET_REF` 兼容路径。
- 监控源的 `secret_ref`、`webhook_secret_ref`、`tls_profile_ref` 暂不改造，仍由现有
  Secret Provider 处理。
- 凭据明文只允许存在于一次请求和数据库驱动调用期间，不进入日志、事件、Artifact、
  幂等结果、异常、Trace 或重试载荷。
- 数据库凭据加密密钥不写入数据库，不通过 Portal 配置，不复用认证密钥。
- Target、凭据和内部发放请求必须进行 Domain、Target、凭据类型和状态校验。
- 所有新代码完成后统一删除旧字段、旧 DTO、旧测试夹具和旧 OpenAPI 定义。

## 3. 阶段一：完成加密基础设施

### 工作项

1. 新增 AIOps 专用 `CredentialCipher`。
2. 使用 AES-256-GCM：每个用户名和密码独立随机 12 字节 nonce。
3. AAD 固定包含：`domain_id`、`credential_id`、`credential_kind`。
4. 严格校验 `KBOT_AIOPS_CREDENTIAL_ENCRYPTION_KEY` 为 Base64URL 编码的 32 字节密钥。
5. 接入 `KBOT_AIOPS_CREDENTIAL_KEY_VERSION`。
6. 统一处理错误：密文损坏、密钥版本不支持和认证标签错误不得泄露细节。
7. 在平台设置派生逻辑中加入用途隔离密钥，但不自动复用认证用途密钥。

### 文件范围

- `packages/platform_core/src/platform_core/config/settings.py`
- `services/aiops_agent/src/aiops_agent/config.py`
- `services/aiops_agent/src/aiops_agent/application/credential_cipher.py`
- `packages/platform_core/src/platform_core/security/`

### 验收

- 同一明文两次加密结果不同。
- 修改密文任意一个 bit 后解密失败。
- 错误密钥、错误版本和错误 Domain 均不能解密。
- 测试输出和日志中不出现用户名或密码。

## 4. 阶段二：完成持久化边界

### 工作项

1. 完善 `CredentialEntity`，包括状态、密钥版本、审计字段和 Domain 约束。
2. 新增 `CredentialRepository`：创建、按 Domain 查询、按 ID+类型读取、撤销。
3. 在 `AIOpsUnitOfWork` 暴露 `credentials` Repository。
4. Target Repository 只接收凭据 ID，不接收密文或明文。
5. 校验凭据类型和 Target Domain 必须一致。
6. 让事务同时提交 Credential 和 Target，失败时整体回滚。

### 文件范围

- `services/aiops_agent/src/aiops_agent/entities/credential.py`
- `services/aiops_agent/src/aiops_agent/entities/__init__.py`
- `services/aiops_agent/src/aiops_agent/repositories/credential.py`
- `services/aiops_agent/src/aiops_agent/repositories/__init__.py`
- `services/aiops_agent/src/aiops_agent/persistence/uow.py`
- `services/aiops_agent/src/aiops_agent/entities/target.py`

### 验收

- Repository 不调用 Application 服务，不调用外部 HTTP。
- 不允许跨 Domain 读取或写入凭据。
- 撤销凭据后不能产生新的诊断或变更授权。

## 5. 阶段三：完成 Target 配置 API

### 请求契约

创建/轮换时允许：

```json
{
  "diagnostic_credential": {
    "username": "kbot_diag",
    "password": "一次性提交"
  },
  "execution_credential": {
    "username": "kbot_exec",
    "password": "一次性提交"
  }
}
```

详情只返回：

```json
{
  "configured": true,
  "credential_id": "UUIDv7",
  "key_version": "2026-08",
  "updated_at": "UTC 时间"
}
```

### 工作项

1. `TargetCreate` 接收凭据输入并禁止额外字段。
2. `TargetPatch` 不接受密码；普通编辑不改变凭据。
3. 新增：
   - `POST /targets/{target_id}/diagnostic-credential:rotate`
   - `POST /targets/{target_id}/execution-credential:rotate`
   - `POST /targets/{target_id}/execution-credential:remove`
4. 轮换命令要求 `If-Match` 与 `Idempotency-Key`。
5. 诊断凭据轮换使 Target 回到 `MAINTENANCE`，并重置健康状态。
6. 执行凭据移除只关闭变更执行，不影响只读诊断。
7. `_idempotent` 对含密码请求使用 HMAC 摘要，禁止保存原始 SHA-256 密码摘要。

### 文件范围

- `packages/platform_core/src/platform_core/contracts/aiops/configuration.py`
- `services/main_api/src/main_api/api/ops.py`
- `services/aiops_agent/src/aiops_agent/api/management/routes.py`
- `services/aiops_agent/src/aiops_agent/application/configuration/target_service.py`
- `services/aiops_agent/src/aiops_agent/application/configuration/projections.py`
- `packages/platform_clients/src/platform_clients/aiops.py`

## 6. 阶段四：完成运行时和 Executor

### 工作项

1. `DiagnosticExecutionGrant` 只包含 `diagnostic_credential_id`。
2. `MutationExecutionGrant` 只包含 `execution_credential_id`。
3. 删除 Executor 对 `ConfiguredSecretStore.resolve()` 的 Target 凭据调用。
4. 新增受保护内部凭据发放端点：
   - 仅允许 DB Executor 服务身份。
   - 接收短时、一次性、audience 绑定的 Grant。
   - 校验 Target、Domain、凭据类型、状态、Target row version 和过期时间。
   - 返回一次性用户名/密码响应。
5. Executor 只在数据库驱动调用期间保留明文引用。
6. 诊断、变更、重试、错误和回调中均不得保存明文。
7. 运行快照只保存 Credential ID 和配置状态，不保存凭据内容。

### 文件范围

- `packages/platform_core/src/platform_core/contracts/aiops/executor.py`
- `services/aiops_agent/src/aiops_agent/application/runtime/service.py`
- `services/aiops_agent/src/aiops_agent/application/changes/service.py`
- `services/aiops_agent/src/aiops_agent/workers/database_handlers.py`
- `services/aiops_agent/src/aiops_agent/executor/service.py`
- `services/aiops_agent/src/aiops_agent/executor/mutation_service.py`
- `services/aiops_agent/src/aiops_agent/adapters/aiops_execution_client.py`
- `services/aiops_agent/src/aiops_agent/api/executions/routes.py`
- `services/aiops_agent/src/aiops_agent/bootstrap/api.py`
- `services/aiops_agent/src/aiops_agent/bootstrap/executor.py`

## 7. 阶段五：同步数据库规范

### Canonical DDL

更新：

- `database/oracle/aiops_agent/001_ops_roots.sql`
- `database/oracle/aiops_agent/006_ops_fks_views.sql`
- `database/oracle/aiops_agent/schema_manifest.json`
- `tests/acceptance/check_oracle_schema.py`
- `tests/acceptance/check_aiops_entity_schema.py`

Canonical DDL 必须包含 `KBOT_OPS_CREDENTIAL`，Target 只保留两个 Credential ID 外键，
不得再次出现 `DIAGNOSTIC_SECRET_REF`、`EXECUTION_SECRET_REF`。

### 已部署开发环境

数据库脚本已执行的环境只需确认最终结构，不再执行旧引用迁移。新 Target 必须通过新
页面重新录入凭据；不提供旧 `env://` 自动导入。

## 8. 阶段六：契约、测试与文档

1. 重生成：
   - `docs/openapi/aiops_public_v1.json`
   - `docs/openapi/aiops_internal_v1.json`
   - `docs/openapi/main_api_public_v1.json`
2. 删除所有 Target 旧 SecretRef 测试夹具和 Smoke 配置。
3. 新增测试：
   - 加密随机性和篡改检测。
   - 凭据不回显。
   - 跨 Domain 拒绝。
   - 凭据类型错配拒绝。
   - 轮换并发冲突。
   - Grant 重放、过期和错误 audience 拒绝。
   - 日志/事件/幂等记录不含凭据。
4. 运行：

```bash
python3 -m compileall -q packages services tests
python3 tests/acceptance/check_oracle_schema.py
python3 tests/acceptance/check_aiops_entity_schema.py
python3 -m pytest tests/unit/aiops_agent -q
```

5. 对旧字段执行最终残留扫描：

```bash
rg -n "diagnostic_secret_ref|execution_secret_ref|AIOPS_DEMO_ORACLE_READONLY" \
  packages services tests docs configuration
```

结果必须为空；监控源的 `secret_ref` 不属于该扫描目标。

## 9. 最终完成标准

- 新 Target 可以从页面直接提交用户名和密码并成功加密保存。
- Target 详情、列表、日志、事件、审计和 OpenAPI 均不泄露密码。
- 诊断和变更都能通过 Credential ID 完成一次受控数据库连接。
- Executor 不读取旧 SecretRef，不持有数据库 Schema 凭据。
- 数据库 canonical DDL、已部署脚本和 ORM 完全一致。
- 编译、契约、Schema、单元和定向集成测试全部通过。
