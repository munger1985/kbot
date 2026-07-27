# 运维与发布脚本

本目录只保留部署、初始化、安全配置和发布流程会直接使用的脚本，并按用途分组：

- `db/apply_oracle_schema.py`：按 `init_services.ini` 初始化所选服务的 Oracle 表。
- `deployment/check_deployment.py`：启动前检查部署配置与生产 Secret。
- `deployment/install_workspace.sh`：安装锁定依赖和七个可编辑 Python 包。
- `security/generate_portal_api_key.py`：生成 Portal API Key 及配置摘要。
- `release/verify_release.py`：编排发布前检查、实库验收并生成发布证据。

静态契约检查、实库 Smoke 和质量评测属于测试资产，统一放在 `tests/`。
