# 运维与发布脚本

本目录只保留部署、初始化、安全配置和发布流程会直接使用的脚本，并按用途分组：

- `scripts/db/apply_oracle_schema.py`：按同目录 `init_services.ini` 初始化所选服务的 Oracle 表。
- `scripts/db/apply_notification_schema.py`：为现有 Oracle Schema 补建并核验通知中心表。
- `scripts/db/init_services.ini`：选择本次初始化包含的业务服务；基础共享表始终创建。
- `scripts/db/align_ammolite_aiops_knowledge_schema.sql`：只补齐 AIOps、知识检索、权限与 Data Query 表结构、字段、索引、外键和视图，不迁移历史数据、不写权限种子、不删除旧业务表。
- `scripts/db/seed_aiops_knowledge_roles_permissions.sql`：幂等补齐 AIOps 与知识检索默认权限、角色和角色权限映射，不创建用户、不分配成员角色。
- `scripts/db/repair_model_serving_s4.py`：定向补齐已有模型目录的行版本字段。
- `scripts/deployment/check_deployment.py`：启动前检查部署配置与生产 Secret。
- `scripts/deployment/install_workspace.sh`：开发环境安装第三方依赖和内部 editable package；生产环境构建并安装 Wheel。
- `scripts/deployment/models/`：按需下载本地 OCR、Tokenizer、VLM 或视觉模型。
- `scripts/security/generate_portal_api_key.py`：生成 Portal API Key 及配置摘要。
- `scripts/release/verify_release.py`：编排发布前检查、实库验收并生成发布证据。

静态契约检查、实库 Smoke 和质量评测属于测试资产，统一放在 `tests/`。
