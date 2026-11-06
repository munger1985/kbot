"""资源采集可无 Agent，但对话创建仍要求 Agent；只做离线 Schema/合同验收。"""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import oracle, postgresql
from pydantic import ValidationError

from aiops_agent.entities import OpsRunEntity
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.workers.task_worker import AIOpsTaskWorker
from platform_core.contracts.aiops import TaskLease
from platform_core.contracts.aiops.public import OpsRunCreate
from platform_core.identity import uuid7


TENANT_SCOPED = hasattr(OpsRunEntity, "tenant_id")
ROOT = Path(__file__).resolve().parents[3]


def resource_lease():
    now = datetime.now(UTC)
    return TaskLease(
        task_id=uuid7(), run_id=uuid7(), task_key="scope", task_type="CONTEXT_BUILD",
        handler_id="database.scope", handler_version="1", input_schema_version="RUN_INPUT.v1",
        output_schema_version="DATABASE_SCOPE_RESULT.v1", lease_token=uuid7(),
        lease_until=now + timedelta(seconds=30), attempt=1, max_attempts=1,
        timeout_seconds=30, row_version=1, target_id=uuid7(), agent_id=None, actor_id=str(uuid7()),
        trace_id="resource-collection", original_request="周期快照",
        plan_snapshot={"trigger": {"type": "SCHEDULE"}, "workload_collection": {"kind": "WORKLOAD"}},
    )


def test_nullable_agent_survives_worker_context_and_run_summary():
    lease = resource_lease()
    assert lease.agent_id is None
    assert AIOpsTaskWorker._context(lease).agent_id is None
    run = SimpleNamespace(
        ops_run_id=lease.run_id, agent_id=None, target_id=lease.target_id,
        trigger_type="SCHEDULE", interaction_mode="AUTONOMOUS", workflow_kind="RESOURCE_COLLECTION",
        status="CREATED", row_version=1, created_at=datetime.now(UTC), completed_at=None,
    )
    assert AIOpsRuntimeService._run_summary(run).agent_id is None
    with pytest.raises(ValidationError):
        OpsRunCreate.model_validate({"target_id": str(lease.target_id)})


def test_canonical_create_table_allows_agentless_collection_without_other_nullable_changes():
    dialect = postgresql.dialect() if TENANT_SCOPED else oracle.dialect()
    sql = str(CreateTable(OpsRunEntity.__table__).compile(dialect=dialect)).upper()
    for name in ("AGENT_ID", "AGENT_VERSION_ID"):
        line = next(line for line in sql.splitlines() if line.strip().startswith(name + " "))
        assert "NOT NULL" not in line
    assert not OpsRunEntity.__table__.c.actor_id.nullable
    assert not OpsRunEntity.__table__.c.domain_id.nullable


def test_explicit_deployment_aligns_only_resource_collection_agent_columns():
    if TENANT_SCOPED:
        from scripts.deployment.init_postgresql import _align_aiops_resource_collection, initialize
        import inspect
        connection = SimpleNamespace(execute=AsyncMock())
        asyncio.run(_align_aiops_resource_collection(connection, schema="test_app"))
        sql = str(connection.execute.await_args.args[0])
        assert sql.count("DROP NOT NULL") == 2
        assert "agent_id DROP NOT NULL" in sql
        assert "agent_version_id DROP NOT NULL" in sql
        assert "UPDATE " not in sql and "DELETE " not in sql and "DROP TABLE" not in sql
        assert "_align_aiops_resource_collection" in inspect.getsource(initialize)
    else:
        paths = (
            ROOT / "database/oracle/aiops_agent/002_ops_runtime.sql",
            ROOT / "database/oracle/generated/aiops_agent/rebuild_aiops_schema.sql",
            ROOT / "database/oracle/generated/aiops_agent/rebuild_aiops_preserve_sources_targets.sql",
        )
        for path in paths:
            table = path.read_text(encoding="utf-8").split("CREATE TABLE KBOT_OPS_RUN (", 1)[1].split("CREATE INDEX", 1)[0]
            for name in ("AGENT_ID", "AGENT_VERSION_ID"):
                line = next(line for line in table.splitlines() if line.strip().startswith(name + " "))
                assert "NOT NULL" not in line
