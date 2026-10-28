"""从权威执行与原始窗口重放肯定效果，供报告和案例共同复核。"""

from datetime import datetime
from uuid import UUID

from aiops_agent.application.changes.effect import effect_source
from aiops_agent.application.changes.effect_controls import load_control_reports
from aiops_agent.application.effect_measurement import evaluate_effect, validate_window_report
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.contracts.change import ActionVerification, ExecutionResultArtifact
from aiops_agent.contracts.report import ComparisonPlan, ComparisonResult


async def replay_supported_effect(*, uow, run, artifact):
    """只有真正可复算且当前仍获授权的受控执行，才返回已验证效果。"""
    try:
        await effect_source(uow=uow, artifact_id=artifact.artifact_id, run=run, schema="COMPARISON_RESULT.v1")
        result = ComparisonResult.model_validate(artifact.payload_json)
        if str(result.source_run_id) != str(run.ops_run_id) or str(artifact.ops_run_id) != str(run.ops_run_id):
            return None
        target = await uow.targets.get_scoped(target_id=run.target_id, domain_id=run.domain_id)
        version = dict((run.plan_snapshot_json or {}).get("target") or {}).get("row_version")
        active = await uow.agents.get(domain_id=run.domain_id, agent_id=run.agent_id)
        if (target is None or target.status != "ENABLED" or str(target.row_version) != str(version)
            or active is None or active.current_version_id is None
            or not await uow.agents.version_has_target(agent_version_id=active.current_version_id, target_id=run.target_id)):
            return None
        if not (result.measured_effect == "IMPROVED" and result.attribution_strength == "SUPPORTED"
                and result.execution_source == "SOURCE_VERIFIED" and not result.gap_codes
                and result.guardrails_passed is True):
            return None
        plan_artifact, verification_artifact, receipt = [await effect_source(
            uow=uow, artifact_id=identity, run=run, schema=schema,
        ) for identity, schema in (
            (result.comparison_plan_artifact_id, "COMPARISON_PLAN.v1"),
            (result.verification_artifact_id, "ACTION_VERIFICATION.v1"),
            (result.source_result_artifact_id, "EXECUTION_RESULT.v1"),
        )]
        if any(str(item.ops_run_id) != str(run.ops_run_id) for item in (plan_artifact, verification_artifact, receipt)):
            return None
        plan = ComparisonPlan.model_validate(plan_artifact.payload_json)
        action = ActionVerification.model_validate(verification_artifact.payload_json)
        execution_result = ExecutionResultArtifact.model_validate(receipt.payload_json)
        if any(str(value.proposal_id) != str(result.proposal_id) for value in (plan, action, execution_result)):
            return None
        if str(plan.source_run_id) != str(run.ops_run_id) or str(action.source_run_id) != str(run.ops_run_id):
            return None
        if action.status != "VERIFIED" or action.gap_codes or str(action.result_artifact_id) != str(receipt.artifact_id) or execution_result.status != "SUCCEEDED" or result.direct_effect != "ACHIEVED":
            return None
        execution = await uow.changes.get_execution(execution_id=execution_result.execution_id)
        if (execution is None or str(execution.proposal_id) != str(result.proposal_id)
            or str(execution.result_artifact_id) != str(receipt.artifact_id) or execution.status != "SUCCEEDED"
            or execution.proposal_hash != execution_result.proposal_hash or execution.command_hash != execution_result.command_hash):
            return None
        if plan.business_effect is None or plan.business_effect.controlled_comparison is None:
            return None
        frozen = plan.business_effect.model_dump(mode="json")
        occurred = execution_result.occurred_at
        now = await uow.runs.database_now()
        if not datetime.fromisoformat(frozen["frozen_at"]) <= occurred <= now:
            return None
        baseline = await effect_source(uow=uow, artifact_id=plan.business_effect.baseline_artifact_id, run=run, schema="AIOPS_WORKLOAD_REPORT.v2")
        if baseline.content_hash != frozen["baseline_hash"]:
            return None
        control_after_id = result.control_analysis.get("after_artifact_id")
        if not control_after_id:
            return None
        controls = await load_control_reports(
            uow=uow, run=run, control=plan.business_effect.controlled_comparison,
            frozen=frozen, after_artifact_id=UUID(control_after_id),
        )
        known = {str(item.artifact_id): item for item in (plan_artifact, verification_artifact, receipt, baseline, *controls)}
        unknown_refs = [ref for ref in result.measurement_artifact_refs if ref.get("artifact_id") not in known]
        if len(unknown_refs) != 1:
            return None
        after = await effect_source(uow=uow, artifact_id=UUID(unknown_refs[0]["artifact_id"]), run=run, schema="AIOPS_WORKLOAD_REPORT.v2")
        known[str(after.artifact_id)] = after
        if len(result.measurement_artifact_refs) != len(known) or len(set(ref["artifact_id"] for ref in result.measurement_artifact_refs)) != len(known):
            return None
        for ref in result.measurement_artifact_refs:
            source = known[ref["artifact_id"]]
            if source.content_hash != ref["content_hash"] or source.schema_version != ref["schema_version"]:
                return None
        if tuple(ref["content_hash"] for ref in result.measurement_artifact_refs) != result.evidence_hashes:
            return None
        for source in (baseline, after, *controls):
            if validate_window_report(source.payload_json)[1] > now:
                return None
        if validate_window_report(baseline.payload_json) != (plan.baseline_start, plan.baseline_end):
            return None
        if validate_window_report(after.payload_json) != (result.after_start, result.after_end):
            return None
        if (result.baseline_start, result.baseline_end) != (plan.baseline_start, plan.baseline_end):
            return None
        recomputed = evaluate_effect(
            frozen=frozen, baseline=baseline.payload_json, after=after.payload_json,
            occurred_at=occurred, direct_status=action.status, controlled_execution=True,
            baseline_ref=f"artifact:{baseline.artifact_id}", after_ref=f"artifact:{after.artifact_id}",
            control_reports={str(item.artifact_id): item.payload_json for item in controls},
            control_after_ref=str(control_after_id),
        )
        for key in ("measured_effect", "symptom_recovered", "guardrails_passed", "attribution_strength",
                    "measurements", "primary_signals", "guardrail_signals", "load_signals", "control_analysis",
                    "execution_source", "gap_codes", "causal_limitations", "result"):
            # JSON 往返将 tuple 变成 list，通过合同归一化后比较。
            value = recomputed[key]
            if isinstance(value, tuple):
                value = list(value)
            expected = getattr(result, key)
            if isinstance(expected, tuple):
                expected = list(expected)
            if value != expected:
                return None
        if recomputed["attribution_strength"] != "SUPPORTED" or recomputed["gap_codes"]:
            return None
        return result
    except (AIOpsApplicationError, KeyError, TypeError, ValueError, AttributeError):
        return None
