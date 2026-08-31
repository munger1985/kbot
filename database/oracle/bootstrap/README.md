# Oracle Bootstrap 资产

本目录只保存规范 Schema 建好后需要写入的确定性基础数据。SQL 由 `scripts/db/`
操作入口读取和执行，不作为独立迁移脚本使用。

Bootstrap 必须幂等，只能执行 DML 和只读校验；缺表、缺列或约束不符合规范时应立即
终止，不得通过动态 DDL 修复现有 Schema。
