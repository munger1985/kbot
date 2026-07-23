# KBot 4.0 Oracle 建库脚本

本目录是 KBot 4.0 唯一有效的 Oracle Schema 定义。4.0 从空 Schema
全量创建，不读取、转换、回填或保留任何 3.x KBot 表和数据。

脚本按拥有数据的服务拆分：

1. `platform_core/`：平台身份边界和 APEX 公共投影；
2. `model_serving/`：模型目录；
3. `knowledge_core/`：Collection、入库聚合、Evidence、Discovery 和 Relation；
4. `agent_runtime/`：Agent Definition、Run、Task、Artifact、Event 和 Delegation；
5. 后续的 `aiops_agent/` 在对应服务实现时新增。

先按上述服务顺序执行，再按每个目录中文件名前缀顺序执行。数字前缀仅表示
空库建表依赖顺序，不是增量 Migration 版本。应用启动时不得自动执行 DDL，
也不得读取其他服务目录中的表。

当前 4.0 开发阶段修改字段时直接更新所属服务的规范建库脚本，并重新创建测试
Schema。脚本不包含 `DROP`、旧表查询、旧数据导入、兼容视图或回滚逻辑。
执行前运行：

```bash
python3 scripts/check_oracle_schema.py
python3 scripts/apply_oracle_schema.py --dry-run
```

连接用户必须具备当前 Schema 的 `CREATE TABLE`、`CREATE VIEW` 权限，并在专用
应用表空间拥有足够 `QUOTA`。应用表空间必须使用
`SEGMENT SPACE MANAGEMENT AUTO`，Oracle VECTOR 不支持非 ASSM 表空间。不要把
业务表默认创建在 `SYSTEM` 表空间。确认目标是空白 Schema 后执行：

```bash
python3 scripts/apply_oracle_schema.py
```

初始化工具会校验当前 PDB、Schema、已有 KBot 对象、DDL 权限和表空间额度；只要
发现已有 `KBOT_%` 表或视图就会拒绝执行，避免误覆盖现有环境。

数据库初始化完成后，由 Portal/APEX 创建新的 Domain 和业务数据，再通过 4.0
API 入库；禁止从 3.x KBot Schema 复制数据。
