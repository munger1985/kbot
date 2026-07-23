# Repository Guidelines

## Project Structure & Module Organization

KBot is a Python/FastAPI knowledge-base and AIOps backend. Deployable entry points live in `apps/`; service-owned code belongs in bounded packages such as `knowledge_core/` and `model_serving/`. Shared configuration, authentication, logging, database primitives, and contracts live in `platform_core/`; cross-service clients live in `platform_clients/`. Keep each service’s API, application, domain, persistence, and worker code inside that service. Configuration examples are under `configuration/example/`, migrations under `migrations/`, SQL/APEX artifacts under `apex/`, documentation under `docs/`, and tests under `tests/`.

## Build, Test, and Development Commands

Create a Python 3.10 environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run a service with its module entry point, for example `python -m apps.knowledge_core_api.main`. Use the configured local environment when integration dependencies are required.

Tests are currently integration-style scripts that may require configured databases, models, or credentials. Run a targeted check, for example:

```bash
python3 scripts/check_4_0_boundaries.py
python3 scripts/check_kc_migrations.py
```

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` for functions, variables, and modules, and `PascalCase` for classes. New or modified comments, docstrings, and human-readable log messages must use Chinese. Stable API fields, error codes, identifiers, protocol values, and third-party names remain English. Follow the surrounding type hints, async patterns, and `loguru` style. Keep API adapters thin, place use cases in application services, and keep SQLAlchemy access in repositories. Repository methods must not call `commit()`; transaction ownership belongs to the Unit of Work.

KBot 4.0 is a clean-slate release. Do not add compatibility imports, V1 routes, dual-read/write paths, or adapters for 3.x. Obsolete code is deleted and recovered from Git history when needed, not retained in active packages or `legacy/`. The Knowledge Core implemented during 3.5 is the 4.0 KC baseline; extend and harden it instead of creating a parallel implementation.

Product and API versions are independent. Public Main API routes start at `/api/v1`; service-only routes start at `/internal/v1` and must not be exposed externally. Add `v2` only when an incompatible version of the same contract must coexist. Unversioned health probes such as `/healthz` and `/readyz` are allowed.

## Testing Guidelines

Add or update a focused `tests/test_<feature>.py` script alongside behavior changes. Include a runnable `__main__` entry point when the test is intended for direct execution, and keep test data/environment assumptions explicit. Do not commit real OCI keys, database passwords, tokens, or `.env`/secret configuration.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style prefixes, commonly `feat(scope):`, `fix(scope):`, and `fix:`; write concise imperative summaries, for example `feat(search): add graph reranking`. Keep commits scoped. Pull requests should explain the behavior change, identify configuration or schema impacts, list tests run, link related issues, and include request/response examples or screenshots for API/UI-visible changes.
