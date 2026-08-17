# 运维与发布脚本

本目录只保留部署、初始化、安全配置和发布流程会直接使用的脚本，并按用途分组：

- `scripts/db/apply_oracle_schema.py`：按同目录 `init_services.ini` 初始化所选服务的 Oracle 表；`--foundation-only` 可幂等补齐平台基础数据和用户安全等级字段。
- `scripts/db/init_services.ini`：选择本次初始化包含的业务服务；基础共享表始终创建。
- `scripts/db/reset_kbot_schema.sql`：显式删除当前用户下的 `KBOT_%` 表和视图，仅用于确认不保留数据的开发 Schema。
- `scripts/deployment/check_deployment.py`：启动前检查部署配置与生产 Secret。
- `scripts/deployment/install_workspace.sh`：开发环境安装第三方依赖和内部 editable package；生产环境构建并安装 Wheel。
- `scripts/deployment/bootstrap_kbot.sh`：空白环境一键安装、配置检查、建表和系统种子初始化；发现已有 KBot 对象时停止。
- `scripts/deployment/models/`：按需下载本地 OCR、Tokenizer、VLM 或视觉模型。
- `scripts/security/generate_portal_api_key.py`：生成 Portal API Key 及配置摘要。
- `scripts/release/verify_release.py`：编排发布前检查、实库验收并生成发布证据。

静态契约检查、实库 Smoke 和质量评测属于测试资产，统一放在 `tests/`。

空白环境部署前先准备 `configuration/kbot.toml` 和 Secret，然后执行：

```bash
bash scripts/deployment/bootstrap_kbot.sh --production
```

该入口不会清理已有 Schema。它会初始化默认业务域 `default`、完整 App 权限与角色
模板、`ADMIN` 凭据、最高安全等级和默认业务域中的 `system_admin` 授权，并写入版本化 Prompt
Catalog；不会创建模型、Collection、Agent、会话或 AIOps 业务数据。

KBot 4.0 不在 `scripts/` 保留一次性升级、修复或补种脚本。Schema 变化直接更新
`database/oracle/` 的规范 DDL，并通过新的空白 Schema 重新初始化。
