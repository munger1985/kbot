# KBot 4.0

KBot 是面向知识检索与数据库运维分析的 Python/FastAPI 后端。4.0 采用单仓库、
同一 Oracle/APEX Schema 下的独立服务架构，不兼容 3.x API、表模型或 Agent
运行时。

## 当前服务

- `main_api/`：面向 Portal 的唯一公开 API/BFF，只暴露 `/api/v1/*`。
- `knowledge_core/`：Collection、Bundle、Document、解析、索引与两阶段检索。
- `model_serving/`：LLM、Embedding、VLM 和 Visual 模型托管。
- `agent_runtime/`：Agent 状态机、计划、Skill 契约和持久化执行内核。
- `platform_core/`：配置、日志、认证、数据库运行时和共享契约。
- `platform_clients/`：跨服务客户端。
- `apps/`：各独立进程入口。

Agent Runtime 当前已建立领域契约、全量 Schema 和内部 API 进程骨架；Run
命令、Repository/UoW 与 Worker 尚未启用，因此不会暴露占位成功接口。

Portal 使用预配置 API Key 访问 Main API；Main API 校验 Domain 后，为内部调用签发
短期 AuthContext JWT。`/internal/v1/*` 仅供服务间调用，不通过 Main API 暴露。

Knowledge Core 来源于 3.5 已完成的实现，是 4.0 的正式基线，不存在平行的旧
知识库运行链路。

## 本地环境

使用 Python 3.10 创建环境并安装依赖：

```bash
pip install -r requirements.txt
```

准备 `configuration/base.toml`、环境配置和 `.env`，不要把密码、Token 或私钥
提交到仓库。配置模板位于 `configuration/example/`。

可单独启动服务：

```bash
python3 -m apps.main_api.main
python3 -m apps.knowledge_core_api.main
python3 -m apps.knowledge_core_parser.main
python3 -m apps.knowledge_core_projection.main
python3 -m apps.ai_models_embedding.main
```

本地同时启动或停止当前服务：

```bash
bash start_kbot.sh
bash stop_kbot.sh
```

## 开发检查

```bash
python3 scripts/check_4_0_boundaries.py
python3 scripts/check_oracle_schema.py
python3 -m unittest discover -s tests
```

完整架构和实施计划见
[`docs/kbot_4.0_design/README.md`](docs/kbot_4.0_design/README.md)。贡献规则见
[`AGENTS.md`](AGENTS.md)。
