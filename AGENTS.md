# Repository Guidelines

## Project Structure & Module Organization

KBot is a Python/FastAPI knowledge-base and AIOps backend. Independently buildable services live under `services/<service>/src/<package>/`; each service owns its entry points, API, application, domain, persistence, workers, and private resources. Shared configuration, authentication, logging, database primitives, and contracts live in `packages/platform_core`; cross-service clients live in `packages/platform_clients`. Configuration examples are under `configuration/`, SQL artifacts under `database/`, documentation under `docs/`, developer pages under `tools/dev_console/`, and all automated checks, Smoke programs, and quality evaluation tools under `tests/`. Keep `scripts/` for deployment, initialization, provisioning, and release operations only.

## Build, Test, and Development Commands

Create a Python 3.10 environment and install dependencies:

```bash
bash scripts/deployment/install_workspace.sh
```

Run a service with its module entry point, for example `python -m knowledge_core.entrypoints.api`. Use the configured local environment when integration dependencies are required.

Tests are currently integration-style scripts that may require configured databases, models, or credentials. Run a targeted check, for example:

```bash
python3 tests/acceptance/check_4_0_boundaries.py
python3 tests/acceptance/check_oracle_schema.py
```

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` for functions, variables, and modules, and `PascalCase` for classes. New or modified comments, docstrings, and human-readable log messages must use Chinese. Stable API fields, error codes, identifiers, protocol values, and third-party names remain English. Follow the surrounding type hints, async patterns, and `loguru` style. Keep API adapters thin, place use cases in application services, and keep SQLAlchemy access in repositories. Repository methods must not call `commit()`; transaction ownership belongs to the Unit of Work.

KBot 4.0 is a clean-slate release. Do not add compatibility imports, V1 routes, dual-read/write paths, or adapters for 3.x. Obsolete code is deleted and recovered from Git history when needed, not retained in active packages or `legacy/`. The Knowledge Core implemented during 3.5 is the 4.0 KC baseline; extend and harden it instead of creating a parallel implementation.

Product and API versions are independent. Public Main API routes start at `/api/v1`; service-only routes start at `/internal/v1` and must not be exposed externally. Add `v2` only when an incompatible version of the same contract must coexist. Unversioned health probes such as `/healthz` and `/readyz` are allowed.

Public routes authenticate the Portal backend API Key and derive Domain/user context from trusted headers. Internal routes require both the service credential and an audience-bound short-lived AuthContext JWT. Never trust caller-supplied actor headers, forward the Portal API Key downstream, or cache an internal JWT in a long-lived HTTP session.

## Testing Guidelines

Add or update focused tests under `tests/unit/<service>/`, `tests/integration/`, or `tests/contract/`. Explicit environment checks belong in `tests/acceptance/`, real dependency flows in `tests/smoke/`, and quality datasets or runners in `tests/evaluation/`. Include a runnable `__main__` entry point when a tool is intended for direct execution, and keep environment assumptions explicit. Do not commit real OCI keys, database passwords, tokens, or `.env`/secret configuration.

## Database Constraint Policy

KBot 业务表只使用主键、外键和列级 `NOT NULL` 保障结构完整性。业务枚举、状态迁移、数值范围、字段组合关系和业务唯一性必须由应用层合同、领域服务及事务处理负责；规范 DDL 不得新增 `CHECK` 或 `UNIQUE` 表约束，也不得用唯一索引绕过这项原则。修改已有表时，不要扩展旧业务约束；应通过无损升级脚本逐步移除，并同步应用校验、测试和数据库文档。

## Runtime Database Schema Policy

运行时代码不得主动校验应用自身数据库 Schema 的版本、表、列、约束、索引或视图是否与代码一致，也不得把这类校验放入启动流程、readiness、Unit of Work、业务请求或模型调用门禁。数据库 readiness 只允许检查连接与最小查询是否可执行（Oracle 使用 `SELECT 1 FROM DUAL`，其他数据库使用等价的 `SELECT 1`）以及真实运行组件的健康状态。Schema 缺失或漂移由正常仓储访问失败暴露，不在运行前重复推断。

应用自身 Schema 的准确性只由 `database/` 下的规范 DDL、初始化/升级脚本、部署流程和 `tests/acceptance/` 校验。修改持久化模型时必须同步这些工件，并运行 `python3 tests/acceptance/check_4_0_boundaries.py`。Data Query 对用户配置的数据源做元数据发现、AIOps 对受运维目标库做能力探测或 DBA 诊断属于产品功能，允许读取外部目标库系统目录；这项例外不得用于校验 KBot 自身持久化 Schema。

## AIOps Synchronization Policy

AIOps App 的领域行为、公共/内部合同、数据库模型、部署工件、测试和用户可见功能必须与 Ammolite 同步维护。若平台基础设施不同，只允许实现层采用各自数据库或运行环境所需的差异；产品语义与能力边界必须保持一致，并在同一项工作中分别验证两个仓库。

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style prefixes, commonly `feat(scope):`, `fix(scope):`, and `fix:`; write concise imperative summaries, for example `feat(search): add graph reranking`. Keep commits scoped. Pull requests should explain the behavior change, identify configuration or schema impacts, list tests run, link related issues, and include request/response examples or screenshots for API/UI-visible changes.
