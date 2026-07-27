# 测试与验收

测试资产按验证层次组织：

- `unit/<service>/`：不依赖真实外部系统的服务单元和组件测试。
- `integration/oracle/`：Oracle Repository、DDL 和 Entity 集成测试。
- `contract/`：配置、OpenAPI、进程拓扑和仓库边界契约。
- `acceptance/check_*.py`：架构、DDL、供应链及外部依赖验收。
- `smoke/`：使用开发 Oracle 或监控服务完成真实链路验收。
- `evaluation/`：使用黄金数据集执行解析和检索质量评测。
- `support/`：只供测试程序复用的辅助模块。

常用离线检查：

```bash
python3 tests/acceptance/check_4_0_boundaries.py
python3 tests/acceptance/check_oracle_schema.py
python3 -m unittest discover -s tests -t .
```
