# 运维与发布脚本

本目录只保留部署、初始化、安全配置和发布流程会直接使用的脚本，并按用途分组：

- `scripts/db/apply_oracle_schema.py`：按同目录 `init_services.ini` 初始化所选服务的 Oracle 表。
- `scripts/db/init_services.ini`：选择本次初始化包含的业务服务；基础共享表始终创建。
- `scripts/deployment/check_deployment.py`：启动前检查部署配置与生产 Secret。
- `scripts/deployment/install_workspace.sh`：安装锁定的第三方依赖；仓库内服务直接从源码加载。
- `scripts/deployment/models/`：按需下载本地 OCR、Tokenizer、VLM 或视觉模型。
- `scripts/security/generate_portal_api_key.py`：生成 Portal API Key 及配置摘要。
- `scripts/release/verify_release.py`：编排发布前检查、实库验收并生成发布证据。

静态契约检查、实库 Smoke 和质量评测属于测试资产，统一放在 `tests/`。