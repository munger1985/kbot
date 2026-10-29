"""诊断只读取已有信息；旧配置、Grant、队列和通用查询均不能恢复主动估算。"""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import get_args
from unittest.mock import AsyncMock

from jose import jwt
import pytest
from pydantic import ValidationError

from aiops_agent.application.diagnosis.sql_plan_presentation import plan_availability_block
from aiops_agent.application.sql_plan_materials import SqlPlanMaterialService
from aiops_agent.application.sql_plan_material_vault import SqlPlanMaterialVault
from aiops_agent.application.sql_plan_capture_vault import SqlPlanCaptureVault
from aiops_agent.diagnostics.dynamic_query import (
    DynamicQueryPolicySnapshot, DynamicQueryRejected, OracleDynamicQueryPolicy,
    PostgreSQLDynamicQueryPolicySnapshot, PostgreSQLDynamicQueryPolicy, MySQLDynamicQueryPolicySnapshot,
)
from aiops_agent.diagnostics.mysql_dynamic_query import MySQLDynamicQueryPolicy
from aiops_agent.diagnostics.grants import DiagnosticGrantCodec, DiagnosticGrantError
from aiops_agent.diagnostics.registry import DiagnosticRegistry
from aiops_agent.workers.handlers import create_runtime_handler_registry
from aiops_agent.workers.task_worker import AIOpsTaskWorker


ROOT = Path(__file__).resolve().parents[3]


def policies():
    return (OracleDynamicQueryPolicy(DynamicQueryPolicySnapshot()),
        PostgreSQLDynamicQueryPolicy(PostgreSQLDynamicQueryPolicySnapshot()),
        MySQLDynamicQueryPolicy(MySQLDynamicQueryPolicySnapshot()))


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("sql", ["EXPLAIN SELECT * FROM mes.t", "EXPLAIN ANALYZE SELECT * FROM mes.t",
    "EXPLAIN PLAN FOR SELECT * FROM mes.t", "PREPARE q AS SELECT * FROM mes.t", "SELECT * FROM mes.t",
    "ANALYZE mes.t", "SET ROLE business_user", "INSERT INTO plan_table SELECT 1", "EXECUTE q"])
def test_dynamic_diagnosis_refuses_business_access_and_plan_requests(index, sql):
    with pytest.raises(DynamicQueryRejected):
        policies()[index].validate(sql)


@pytest.mark.parametrize("index,sql", [(0, "SELECT inst_id, sql_id FROM gv$sql_plan"),
    (1, "SELECT relid, seq_scan FROM pg_catalog.pg_stat_user_tables"),
    (2, "SELECT digest, count_star FROM performance_schema.events_statements_summary_by_digest")])
def test_existing_dictionary_evidence_still_readable(index, sql):
    result = policies()[index].validate(sql)
    assert result.referenced_objects and not result.normalized_sql.upper().startswith("EXPLAIN")


def test_postgresql_policy_has_no_explain_wrapper_argument():
    with pytest.raises(TypeError):
        policies()[1].validate("SELECT 1", explain=True)


def test_historical_services_have_no_submission_or_decryption_for_execution():
    assert not hasattr(SqlPlanMaterialService, "submit")
    assert not hasattr(SqlPlanMaterialVault, "seal")
    assert not hasattr(SqlPlanMaterialVault, "open_for_capture")
    assert not hasattr(SqlPlanCaptureVault, "create")
    assert not hasattr(SqlPlanCaptureVault, "update")
    assert not hasattr(DiagnosticGrantCodec, "issue_plan_capture")
    assert not hasattr(DiagnosticGrantCodec, "verify_plan_capture")


def test_agent_contract_refuses_estimation_policy_and_connection_test_does_not_probe_explain():
    from aiops_agent.application.agents import CreateAIOpsAgentCommand, UpdateAIOpsAgentCommand
    for contract in (CreateAIOpsAgentCommand, UpdateAIOpsAgentCommand):
        field = contract.model_fields.get("execution_policy")
        model = field.annotation if field is not None else contract
        if not hasattr(model, "model_fields"):
            model = next(item for item in get_args(model) if hasattr(item, "model_fields"))
        assert "sql_plan_capture" not in model.model_fields
        payload = {"sql_plan_capture": {"enabled": True, "allowed_objects": ["mes.t"]}}
        if "knowledge_policy" in model.model_fields:
            payload["knowledge_policy"] = {"enabled": False}
        with pytest.raises(ValidationError) as failure:
            model.model_validate(payload)
        assert any(error["loc"] == ("sql_plan_capture",) and error["type"] == "extra_forbidden"
            for error in failure.value.errors())
    candidates = ("application/configuration/connection_test.py", "adapters/target_connection.py")
    for path in candidates:
        source = ROOT / "services/aiops_agent/src/aiops_agent" / path
        if source.exists():
            text = source.read_text()
            assert "EXPLAIN (FORMAT JSON)" not in text
            assert "explain_json" not in text


def test_signed_legacy_plan_grant_is_rejected_by_both_read_channels():
    now = datetime.now(UTC)
    claims = {"iss": "worker", "aud": "executor", "iat": int(now.timestamp()), "nbf": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=30)).timestamp()), "jti": "legacy", "typ": "kbot-sql-plan-capture-grant"}
    codec = DiagnosticGrantCodec(secret="x" * 32, issuer="worker", audience="executor")
    token = jwt.encode(claims, "x" * 32, algorithm="HS256")
    for verify in (codec.verify, codec.verify_dynamic):
        with pytest.raises(DiagnosticGrantError, match="类型无效"):
            verify(token)


def test_registered_routes_and_bootstraps_have_no_active_estimate_path():
    from aiops_agent.api.runtime.routes import router as runtime
    from aiops_agent.api.executions.routes import router as executions
    from aiops_agent.api.executor.routes import router as executor
    from main_api.api.ops import router as public
    for router in (runtime, executions, executor, public):
        for route in router.routes:
            assert "sql-plan-capture" not in route.path
            assert "sql-plan-material-sources" not in route.path
            if route.path.endswith("/sql-plan-materials"):
                assert route.methods == {"GET"}
    for name in ("api", "executor", "worker"):
        text = (ROOT / f"services/aiops_agent/src/aiops_agent/bootstrap/{name}.py").read_text()
        for forbidden in ("SqlPlanCaptureExecutor", "NativeSqlPlanCaptureDriver", "SqlPlanCaptureExecutionService",
            "SqlPlanCaptureService", "exchange_plan_capture", "report_plan_capture"):
            assert forbidden not in text


def test_queued_legacy_capture_fails_without_credentials_or_retry_classification():
    registry = create_runtime_handler_registry()
    assert all(row.handler_id != "sql.plan-capture" for row in registry.manifests)
    from tests.unit.aiops_agent.test_aiops_task_worker import AIOpsTaskWorkerTest
    lease = AIOpsTaskWorkerTest._lease().model_copy(update={"handler_id": "sql.plan-capture",
        "plan_snapshot": {"sql_plan_capture": {"enabled": True}}})
    service = SimpleNamespace(fail_task=AsyncMock())
    worker = AIOpsTaskWorker(runtime_service=service, handler_registry=registry, worker_id="test",
        lease_seconds=30, heartbeat_seconds=10, poll_interval_seconds=1)
    asyncio.run(worker._execute(lease))
    failure = service.fail_task.await_args.args[0]
    assert failure.error_code == "HANDLER_NOT_FOUND"


@pytest.mark.parametrize("engine", ["ORACLE", "POSTGRESQL", "MYSQL"])
def test_missing_plan_is_not_an_estimate_request_or_a_diagnosis_blocker(engine):
    block = plan_availability_block(database_type=engine, bundle={})
    assert block.payload["sql_plan_acquisition"]["read_existing_only"] is True
    assert "不发起估算" in block.payload["markdown"]
    assert "仍依据语句统计、等待与监控" in block.payload["markdown"]
    assert "独立审批" not in block.payload["markdown"]
    assert "controlled_estimate_supported" not in block.payload["sql_plan_acquisition"]


def test_oracle_native_plan_wait_monitor_and_history_tools_remain_registered():
    tools = {tool.definition.tool_id for tool in DiagnosticRegistry.load().tools}
    assert {"db.sql.execution_plan", "db.oracle.statement.xplan", "db.oracle.sql_monitor.executions",
        "db.oracle.statement.ash", "db.postgresql.statement.details", "db.sql.cursor_details"} <= tools


def test_prompt_keeps_seed_version_and_database_optimizer_boundary():
    from platform_core.prompts.catalog import load_prompt_catalog
    prompt = next(row for row in load_prompt_catalog().entries if row.prompt_key == "aiops_agent.sql_diagnosis")
    assert prompt.version == "1.0.0"
    for phrase in ("由数据库和 CBO 负责", "不主动发起估算", "不索取业务表访问权限", "不自动收集统计信息"):
        assert phrase in prompt.content
