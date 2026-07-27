"""在配置的 Oracle 上验收只读目录、Grant、Executor 与结果限界。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiops_agent.adapters.secret_store import ConfiguredSecretStore
from aiops_agent.config import get_aiops_settings
from aiops_agent.diagnostics import (
    create_diagnostic_grant_codec,
    create_diagnostic_registry,
)
from aiops_agent.diagnostics.grants import canonical_sha256
from aiops_agent.executor import DiagnosticExecutorService
from aiops_agent.executor.drivers import OracleDiagnosticDriver
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    DiagnosticExecutionGrant,
    DiagnosticLimits,
    ReadDiagnosticRequest,
)
from platform_core.identity import uuid7


async def main() -> None:
    settings = get_aiops_settings()
    oracle = settings.database.oracle
    secret_name = "AIOPS_DATABASE_DIAGNOSTIC_SMOKE"
    os.environ[secret_name] = json.dumps(
        {
            "username": oracle.username,
            "password": oracle.require_password(),
        }
    )
    registry = create_diagnostic_registry(settings)
    codec = create_diagnostic_grant_codec(settings)
    hard_limits = DiagnosticLimits(
        statement_timeout_seconds=30,
        max_result_rows=500,
        max_result_bytes=2 * 1024 * 1024,
    )
    executor = DiagnosticExecutorService(
        registry=registry,
        grant_codec=codec,
        secret_store=ConfiguredSecretStore(
            provider="environment", allowed_schemes=("env",)
        ),
        drivers=(OracleDiagnosticDriver(),),
        hard_limits=hard_limits,
        concurrency=2,
    )
    target_id = uuid7()
    run_id = uuid7()
    results = []
    try:
        for tool_id in (
            "db.instance.identity",
            "db.session.active",
            "db.session.blocking_chain",
            "db.storage.capacity",
        ):
            tool = registry.resolve(
                tool_id=tool_id,
                tool_version="1.0.0",
                db_type="ORACLE",
                db_version="23ai",
                capabilities={
                    "dynamic_performance_views",
                    "dba_catalog_views",
                },
                entitlements=set(),
            )
            now = datetime.now(UTC)
            grant = DiagnosticExecutionGrant(
                issuer=settings.executor.grant_issuer,
                audience=settings.executor.service_name,
                grant_id=uuid7(),
                issued_at=now,
                expires_at=now + timedelta(seconds=30),
                run_id=run_id,
                task_id=uuid7(),
                lease_token_hash="a" * 64,
                target_id=target_id,
                target_row_version=1,
                db_type="ORACLE",
                connection_profile=DiagnosticConnectionProfile(
                    host=oracle.host,
                    port=oracle.port,
                    service=oracle.service_name,
                    tls_enabled=False,
                ),
                diagnostic_secret_ref=f"env://{secret_name}",
                tool_id=tool.definition.tool_id,
                tool_version=tool.definition.version,
                variant=tool.definition.variant,
                template_sha256=tool.definition.template_sha256,
                parameters_sha256=canonical_sha256({}),
                capability_snapshot_hash=registry.catalog_hash,
                limits=DiagnosticLimits(
                    statement_timeout_seconds=tool.definition.timeout_seconds,
                    max_result_rows=tool.definition.max_rows,
                    max_result_bytes=tool.definition.max_bytes,
                ),
                trace_id=str(uuid7()),
            )
            result = await executor.execute(
                ReadDiagnosticRequest(
                    executor_request_id=uuid7(),
                    grant=codec.issue(grant),
                    parameters={},
                    idempotency_key=f"smoke:{tool_id}",
                )
            )
            if result.status != "SUCCEEDED" or result.observation is None:
                raise RuntimeError(
                    f"{tool_id} 执行失败：{result.error_code}"
                )
            results.append(
                (
                    tool_id,
                    result.observation.row_count,
                    result.observation.truncated,
                )
            )
        print(
            "AIOps 数据库诊断 Oracle Smoke 通过："
            + ", ".join(
                f"{tool_id}[rows={rows},truncated={truncated}]"
                for tool_id, rows, truncated in results
            )
        )
    finally:
        os.environ.pop(secret_name, None)


if __name__ == "__main__":
    asyncio.run(main())
