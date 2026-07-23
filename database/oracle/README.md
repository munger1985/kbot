# KBot 4.0 Oracle 建库脚本

本目录是 KBot 4.0 唯一有效的 Oracle Schema 定义。4.0 从空 Schema
全量创建，不读取、转换、回填或保留任何 3.x KBot 表和数据。

脚本按拥有数据的服务拆分：

1. `platform_core/`：平台身份边界和 APEX 公共投影；
2. `model_serving/`：模型目录；
3. `knowledge_core/`：Collection、入库聚合、Evidence、Discovery 和 Relation；
4. 后续的 `agent_runtime/`、`aiops_agent/` 在对应服务实现时新增。

先按上述服务顺序执行，再按每个目录中文件名前缀顺序执行。数字前缀仅表示
空库建表依赖顺序，不是增量 Migration 版本。应用启动时不得自动执行 DDL，
也不得读取其他服务目录中的表。

当前 4.0 开发阶段修改字段时直接更新所属服务的规范建库脚本，并重新创建测试
Schema。脚本不包含 `DROP`、旧表查询、旧数据导入、兼容视图或回滚逻辑。
执行前运行：

```bash
python3 scripts/check_oracle_schema.py
```

数据库初始化完成后，由 Portal/APEX 创建新的 Domain 和业务数据，再通过 4.0
API 入库；禁止从 3.x KBot Schema 复制数据。
