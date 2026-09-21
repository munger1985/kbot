# Oracle Bootstrap 资产

本目录只保存规范 Schema 建好后需要写入的确定性基础数据。SQL 由 `scripts/db/`
操作入口读取和执行，不作为独立迁移脚本使用。

Bootstrap 必须幂等，只能执行 DML 和只读校验；缺表、缺列或约束不符合规范时应立即
终止，不得通过动态 DDL 修复现有 Schema。

`knowledge_retrieval/initial_admin.sql` 和 `media_studio/initial_admin.sql` 由
`scripts/db/initialize_product_app.py` 调用，分别创建两个 App 的空白引导
Domain 与初始管理员。它们不创建业务 Agent、模型绑定或模型配置；
重复执行不会覆盖已有管理员凭据。

```bash
python scripts/db/initialize_product_app.py knowledge_retrieval
python scripts/db/initialize_product_app.py media_studio
```
