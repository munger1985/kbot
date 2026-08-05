# KBot 后端能力迁移 S0 基线报告

## 1. 基线结论

记录日期：2026-08-04。

S0 已完成工作区成员、内部发行包、开发 editable 安装、生产 wheel 安装和模块来源验收。
服务运行不再依赖 `PYTHONPATH` 或把 `packages/*/src`、`services/*/src` 写入
`sys.path`。新增 Data Query 仅包含可安装的最小包骨架，不包含阶段一业务实现。

当前工作区共有 8 个内部发行包：

1. `kbot-platform-core`；
2. `kbot-platform-clients`；
3. `kbot-model-serving`；
4. `kbot-knowledge-core`；
5. `kbot-agent-runtime`；
6. `kbot-aiops-agent`；
7. `kbot-data-query`；
8. `kbot-main-api`。

启动拓扑保持 14 个进程，其中 12 个 HTTP 进程、2 个 Worker。Oracle 全量建库静态基线
保持 5 个服务、21 个脚本；S0 不新增 Data Query 业务表。

## 2. 环境边界

`cube` 环境已经安装 Ammolite 的同名顶级 Python 包，不能同时作为 Ammolite 与 KBot 的
editable 开发环境，否则 `agent_runtime`、`aiops_agent`、`data_query` 等 import 的来源取决于
editable 安装元数据顺序。S0 验收以 `cube` 的 Python 3.12 和第三方依赖为基础，分别建立
隔离的开发、生产虚拟环境；验收结束后不在 `cube` 中保留 KBot 内部发行包。

KBot 开发环境应独立创建，或明确使用只承载 KBot editable 包的环境。安装入口支持通过
`KBOT_PYTHON` 指定解释器；未指定时选择 `KBOT_CONDA_ENV` 或默认 `kbot4`，并在安装末尾
强制验证每个模块的实际来源。

## 3. 源码路径与标识扫描

- `start_kbot.sh` 和运维脚本不再设置 `PYTHONPATH` 或注入服务源码目录。
- 测试工具中保留的根目录 `sys.path` 仅服务于直接执行脚本时导入 `tests.support`；它们不注入
  `packages/*/src` 或 `services/*/src`，不参与服务运行。
- `packages`、`services`、`database`、`scripts`、`configuration`、`tests` 中没有 Ammolite
  产品标识残留。
- 所有新增发行名、版本、模块、提示和文档资产均使用 KBot 标识。

## 4. S0 验收结果

以下检查通过：

```text
bash -n scripts/deployment/install_workspace.sh
开发模式完整安装与 8 个包来源检查
生产模式完整 wheel 构建、离线安装与 8 个包来源检查
python -m compileall -q packages services tests scripts
tests/acceptance/check_4_0_boundaries.py
tests/acceptance/check_process_topology.py
tests/acceptance/check_configuration_contract.py
tests/acceptance/check_oracle_schema.py
14 个工作区、拓扑、发布和 Knowledge Core 定向契约/单元测试
git diff --check
```

开发模式模块均解析到对应 KBot `src/`；生产模式模块均解析到已安装 wheel，不解析仓库源码。

## 5. 非 S0 基线失败

全量测试基线共执行 450 个测试，存在 10 个失败和 16 个错误，集中在本迁移明确排除的
AIOps 凭据切换、AIOps DDL/实体 Manifest、旧测试夹具以及 OpenAPI Snapshot 漂移。
`check_openapi_contracts.py` 当前还报告 Main API public、AIOps public 和 AIOps internal 三份
Snapshot 不一致。这些问题没有由 S0 引入，也不在 S0 中越界修改；后续阶段验收必须继续区分
新增回归与该基线问题。

## 6. 阶段门状态

S0 阶段门通过。下一步为 S1 Data Query 完整迁移与 MCP/Semantic 双模式问数；尚未开始。
