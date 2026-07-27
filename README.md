# KBot 4.0

KBot 是面向知识检索与数据库运维分析的 Python/FastAPI 后端。4.0 采用单仓库、
同一 Oracle/APEX Schema 下的独立服务架构，不兼容 3.x API、表模型或 Agent
运行时。

## 当前结构

- `services/`：五个可独立构建服务及其进程入口。
- `packages/`：`platform_core` 和 `platform_clients` 两个共享 Python 包。
- `database/oracle/`：同一 Schema 下按服务所有权拆分的全量 DDL。
- `configuration/`：唯一部署配置样例。
- `resources/`：部署进程拓扑等不可变运行资源。
- `tests/`：单元、集成、契约、验收、Smoke 和质量评估。
- `tools/dev_console/`：仅开发环境启用的功能测试页面。
- `var/`：本地日志、上传文件和生成物；不进入 Git。

Agent Runtime 已启用持久化 Run/Task/Artifact/Event、固定 Document Plan、
KC 两阶段检索 Skill、Grounded Response Composer、租约恢复和独立 Worker。
Portal 可通过 Main API 的 `/api/v1/agents` 与 `/api/v1/runs` 使用该链路；
内部 Plan、Task Claim 和 Artifact 写回不会公开。

Portal 使用预配置 API Key 访问 Main API；Main API 校验 Domain 后，为内部调用签发
短期 AuthContext JWT。`/internal/v1/*` 仅供服务间调用，不通过 Main API 暴露。

Knowledge Core 来源于 3.5 已完成的实现，是 4.0 的正式基线，不存在平行的旧
知识库运行链路。

## 本地环境

使用 Python 3.10 创建环境并安装依赖：

```bash
bash scripts/deployment/install_workspace.sh
```

脚本先安装锁定依赖，再以 editable 方式安装两个共享包和五个服务包。

按 [configuration/README.md](configuration/README.md) 从
`configuration/kbot.toml.example` 准备唯一部署文件。密码、Token、模型厂商
Key和私钥只能由环境变量或Secret管理系统注入。

可单独启动服务：

```bash
python3 -m main_api.entrypoints.api
python3 -m agent_runtime.entrypoints.api
python3 -m agent_runtime.entrypoints.worker
python3 -m aiops_agent.entrypoints.api
python3 -m aiops_agent.entrypoints.worker
python3 -m aiops_agent.entrypoints.scheduler
python3 -m aiops_agent.entrypoints.db_executor
python3 -m knowledge_core.entrypoints.api
python3 -m knowledge_core.entrypoints.parser
python3 -m knowledge_core.entrypoints.projection
python3 -m model_serving.entrypoints.embedding
```

本地同时启动或停止当前服务：

```bash
bash start_kbot.sh
bash stop_kbot.sh
```

## 开发检查

```bash
python3 tests/acceptance/check_4_0_boundaries.py
python3 tests/acceptance/check_oracle_schema.py
python3 -m unittest discover -s tests -t .
```

当前架构、产品和部署文档统一从
[`docs/README.md`](docs/README.md) 进入。贡献规则见 [`AGENTS.md`](AGENTS.md)，
物理目录说明见
[`docs/architecture/repository-layout.md`](docs/architecture/repository-layout.md)。
