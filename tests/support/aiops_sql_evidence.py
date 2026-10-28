"""构造正式工具来源及其生产投影，禁止用充分性副本替代原始来源夹具。"""

from types import SimpleNamespace
from uuid import UUID

from aiops_agent.application.diagnosis.evidence_projection import database_evidence_fact
from aiops_agent.contracts.tool_execution import DbaToolResult, ToolOutcome
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.executor import DatabaseObservation
from platform_core.identity import uuid7


def backed_sql_fact(fact, run):
    observation = DatabaseObservation.model_validate({
        "executor_request_id": uuid7(), "target_id": run.target_id, "tool_id": fact.tool_id,
        "tool_version": "1.0.0", "variant": "test-original", "template_sha256": "a" * 64,
        "db_type": run.plan_snapshot_json["target"]["db_type"], "db_version": "16",
        "capability_snapshot_hash": "b" * 64, "captured_at": fact.captured_at, "duration_ms": 1,
        "columns": [{**column, "sensitivity": "PUBLIC"} for column in fact.columns],
        "rows": fact.rows, "row_count": fact.row_count, "truncated": fact.truncated,
        "warnings": fact.warnings, "parameters_sha256": "d" * 64,
        "result_sha256": canonical_hash(fact.model_dump(mode="json")["rows"]),
    })
    outcome = ToolOutcome(step_id=fact.step_id, tool_id=fact.tool_id, tool_version="1.0.0",
        status="SUCCEEDED", observation=observation)
    result = DbaToolResult(source_type="TOOL", source_id=fact.source_id, source_version="1.0.0",
        definition_hash="e" * 64, output_schema="DBA_TOOL_RESULT.v1", measurement_semantics=fact.measurement_semantics,
        presentation_kind=fact.presentation_kind, status="SUCCEEDED", tool_outcomes=(outcome,))
    payload = result.model_dump(mode="json")
    artifact = SimpleNamespace(artifact_id=UUID(str(fact.artifact_id)), ops_run_id=run.ops_run_id,
        schema_version="DBA_TOOL_RESULT.v1", trust_level="SOURCE_VERIFIED", payload_json=payload,
        content_hash=canonical_hash(payload), security_level=2)
    projected = database_evidence_fact(result=result, outcome=outcome, artifact_id=artifact.artifact_id)
    return artifact, projected
