# KBot 4.0 Oracle 建库脚本

本目录是 KBot 4.0 唯一有效的 Oracle Schema 定义。4.0 从空 Schema
全量创建，不读取、转换、回填或保留任何 3.x KBot 表和数据。

脚本按拥有数据的服务拆分：

1. `platform_core/`：平台身份边界和 APEX 公共投影；
2. `main_api/`：公开 API 的用户、角色和权限；
3. `model_serving/`：模型目录；
4. `knowledge_core/`：Collection、入库聚合、Evidence、Discovery 和 Relation；
5. `knowledge_retrieval_app/`：知识检索私有 Agent、版本和授权；
6. `km_asset_app/`：KM 资产、Agent，以及 Slack Inbox、会话映射和 Outbox；
7. `agent_runtime/`：Execution Spec 快照、Run、Task、Artifact、Event、会话和记忆；
8. `data_query/`：数据源、语义模型、查询策略和查询运行；
9. `aiops_agent/`：私有 Agent、目标、监控、会话、证据、HITL、执行、巡检和报告。

`platform_core` 是每次初始化都必须创建的基础层，不需要配置。其余已实现服务在
`scripts/db/init_services.ini` 的 `[services]` 中使用 `true`/`false` 选择。
初始化工具先执行
基础层，再按上述业务服务依赖顺序和目录内文件名前缀执行。数字前缀仅表示空库
建表依赖顺序，不是增量 Migration 版本。应用启动时不得自动执行 DDL，也不得读取
其他服务目录中的表。

空白环境推荐从仓库根目录使用统一部署入口：

```bash
bash scripts/deployment/bootstrap_kbot.sh --production
```

它会安装依赖、执行配置与静态契约检查、确认目标为空 Schema、创建全部已选服务
对象，并初始化默认 App 角色/权限/映射和 Prompt Catalog。它不会自动清库，也不会
创建 Domain、用户、成员关系、业务 Agent、模型或知识库数据。

`aiops_agent` 已提供八段规范 DDL 和受控 APEX 投影，可像其他业务服务一样在
初始化配置中选择。其脚本必须整体启用或禁用，不能跳过中间依赖段。

当前 4.0 开发阶段修改字段时直接更新所属服务的规范建库脚本，并重新创建测试
Schema。脚本不包含 `DROP`、旧表查询、旧数据导入、兼容视图或回滚逻辑。
已有开发 Schema 无需保留数据时，先使用 KBot Schema 用户执行
`scripts/db/reset_kbot_schema.sql`，确认验证查询返回 0 行，再运行下方初始化
命令。重置脚本只处理当前用户下的 `KBOT_%` 表和视图。KBot 4.0 不保留一次性
升级、字段修复或数据补种脚本；结构变化直接更新规范 DDL，并在新的空白 Schema
重新初始化。

执行前运行：

```bash
python3 tests/acceptance/check_oracle_schema.py
python3 scripts/db/apply_oracle_schema.py \
  --config scripts/db/init_services.ini \
  --dry-run
```

连接用户必须具备当前 Schema 的 `CREATE TABLE`、`CREATE VIEW` 权限，并在专用
应用表空间拥有足够 `QUOTA`。应用表空间必须使用
`SEGMENT SPACE MANAGEMENT AUTO`，Oracle VECTOR 不支持非 ASSM 表空间。不要把
业务表默认创建在 `SYSTEM` 表空间。确认目标是空白 Schema 后执行：

Knowledge Core 默认通过 `DBMS_ALERT` 唤醒异步 Worker。PDB 管理员还需执行
以下运行权限授权，其中用户名替换为实际 KBot Schema：

```sql
GRANT EXECUTE ON SYS.DBMS_ALERT TO KBOTDEV;
```

```bash
python3 scripts/db/apply_oracle_schema.py \
  --config scripts/db/init_services.ini
```

初始化工具会校验当前 PDB、Schema、已有 KBot 对象、DDL 权限和表空间额度；只要
发现已有 `KBOT_%` 表或视图就会拒绝执行，避免误覆盖现有环境。服务选择用于一次
性部署，不能先初始化部分服务，再对同一 Schema 补跑其他服务；需要变更范围时应
准备新的空白 Schema 后重新执行。

数据库初始化完成后，由 Portal/APEX 创建新的 Domain 和业务数据，再通过 4.0
API 入库；禁止从 3.x KBot Schema 复制数据。
