"""从冻结计划、真实动作来源和原始窗口重放效果；不只核验成功结果。"""

from datetime import datetime
from uuid import UUID

from aiops_agent.application.changes.effect import effect_source
from aiops_agent.application.changes.effect_controls import load_control_reports
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.effect_measurement import apply_action_verification, evaluate_effect, validate_window_report
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.source_authorization import source_scope_is_current
from aiops_agent.contracts.change import ActionVerification, AdvisoryActionResult, ExecutionResultArtifact
from aiops_agent.contracts.report import ComparisonPlan, ComparisonResult


def is_supported_effect(result):
    """肯定验证门槛不因支持失败经验或人工反馈而放宽。"""
    return (result.measured_effect == "IMPROVED" and result.attribution_strength == "SUPPORTED"
        and result.execution_source == "SOURCE_VERIFIED" and result.direct_effect == "ACHIEVED"
        and not result.gap_codes and result.guardrails_passed is True)


async def replay_supported_effect(*, uow, run, artifact):
    """保留严格成功入口，普通效果重放成功不等于处置效果已验证。"""
    result = await replay_effect(uow=uow, run=run, artifact=artifact)
    return result if result is not None and is_supported_effect(result) else None


async def replay_effect(*, uow, run, artifact):
    """复核全部已生产的效果结果，保留人工来源、无效、退化和不可判断。"""
    try:
        artifact = await effect_source(uow=uow, artifact_id=artifact.artifact_id, run=run, schema="COMPARISON_RESULT.v1")
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
        plan_artifact = await effect_source(uow=uow, artifact_id=result.comparison_plan_artifact_id, run=run, schema="COMPARISON_PLAN.v1")
        verification_artifact = await effect_source(uow=uow, artifact_id=result.verification_artifact_id, run=run, schema="ACTION_VERIFICATION.v1")
        receipt = await uow.runs.get_artifact(artifact_id=result.source_result_artifact_id)
        if receipt is None or any(str(item.ops_run_id) != str(run.ops_run_id) for item in (plan_artifact, verification_artifact, receipt)):
            return None
        plan = ComparisonPlan.model_validate(plan_artifact.payload_json)
        action = ActionVerification.model_validate(verification_artifact.payload_json)
        if (plan.business_effect is None or plan.result_rule_version != "business-effect.v1"
            or str(plan.source_run_id) != str(run.ops_run_id) or str(action.source_run_id) != str(run.ops_run_id)
            or any(str(value.proposal_id) != str(result.proposal_id) for value in (plan, action))
            or str(action.result_artifact_id) != str(receipt.artifact_id)):
            return None
        controlled = receipt.schema_version == "EXECUTION_RESULT.v1" and receipt.trust_level == "SOURCE_VERIFIED"
        manual = receipt.schema_version == "USER_PROVIDED_ACTION_RESULT.v1" and receipt.trust_level == "USER_PROVIDED"
        if controlled:
            await effect_source(uow=uow, artifact_id=receipt.artifact_id, run=run, schema="EXECUTION_RESULT.v1")
            execution_result = ExecutionResultArtifact.model_validate(receipt.payload_json)
            execution = await uow.changes.get_execution(execution_id=execution_result.execution_id)
            if (execution_result.status != "SUCCEEDED" or str(execution_result.proposal_id) != str(result.proposal_id)
                or execution is None or str(execution.proposal_id) != str(result.proposal_id)
                or str(execution.result_artifact_id) != str(receipt.artifact_id) or execution.status != "SUCCEEDED"
                or execution.proposal_hash != execution_result.proposal_hash or execution.command_hash != execution_result.command_hash):
                return None
            occurred = execution_result.occurred_at
        elif manual:
            if (sha256_json(receipt.payload_json) != receipt.content_hash
                or not await source_scope_is_current(uow=uow, run=run, domain_id=run.domain_id)):
                return None
            reported = AdvisoryActionResult.model_validate(receipt.payload_json)
            if (reported.status != "EXECUTED" or str(reported.proposal_id) != str(result.proposal_id)
                or str(reported.submitted_by) != str(run.actor_id)):
                return None
            occurred = reported.occurred_at
        else:
            return None
        direct_effect = {"VERIFIED": "ACHIEVED", "NOT_ACHIEVED": "NOT_ACHIEVED", "ADVERSE": "ADVERSE"}.get(action.status, "UNKNOWN")
        if result.direct_effect != direct_effect:
            return None
        frozen = plan.business_effect.model_dump(mode="json")
        now = await uow.runs.database_now()
        if not datetime.fromisoformat(frozen["frozen_at"]) <= occurred <= now:
            return None
        if manual and not occurred <= reported.submitted_at <= now:
            return None
        baseline = await effect_source(uow=uow, artifact_id=plan.business_effect.baseline_artifact_id, run=run, schema="AIOPS_WORKLOAD_REPORT.v2")
        if baseline.content_hash != frozen["baseline_hash"]:
            return None
        controls, control_after_id = (), result.control_analysis.get("after_artifact_id")
        if plan.business_effect.controlled_comparison is not None:
            if not control_after_id:
                return None
            controls = await load_control_reports(
                uow=uow, run=run, control=plan.business_effect.controlled_comparison,
                frozen=frozen, after_artifact_id=UUID(control_after_id),
            )
        elif result.control_analysis:
            return None
        known = {str(item.artifact_id): item for item in (plan_artifact, verification_artifact, receipt, baseline, *controls)}
        unknown_refs = [ref for ref in result.measurement_artifact_refs if ref.get("artifact_id") not in known]
        if len(unknown_refs) != 1:
            return None
        after = await effect_source(uow=uow, artifact_id=UUID(unknown_refs[0]["artifact_id"]), run=run, schema="AIOPS_WORKLOAD_REPORT.v2")
        artifacts = (plan_artifact, baseline, after, verification_artifact, receipt, *controls)
        expected_refs = tuple({"artifact_id": str(item.artifact_id), "content_hash": item.content_hash, "schema_version": item.schema_version} for item in artifacts)
        if (len({str(item.artifact_id) for item in artifacts}) != len(artifacts)
            or result.measurement_artifact_refs != expected_refs
            or result.evidence_hashes != tuple(item.content_hash for item in artifacts)):
            return None
        for source in (baseline, after, *controls):
            if validate_window_report(source.payload_json)[1] > now:
                return None
        if any(str(source.payload_json.get("target_id")) != str(run.target_id) for source in (baseline, after)):
            return None
        if validate_window_report(baseline.payload_json) != (plan.baseline_start, plan.baseline_end):
            return None
        if validate_window_report(after.payload_json) != (result.after_start, result.after_end):
            return None
        if (result.baseline_start, result.baseline_end) != (plan.baseline_start, plan.baseline_end):
            return None
        recomputed = apply_action_verification(evaluate_effect(
            frozen=frozen, baseline=baseline.payload_json, after=after.payload_json,
            occurred_at=occurred, direct_status=action.status, controlled_execution=controlled,
            baseline_ref=f"artifact:{baseline.artifact_id}", after_ref=f"artifact:{after.artifact_id}",
            control_reports={str(item.artifact_id): item.payload_json for item in controls},
            control_after_ref=str(control_after_id) if control_after_id else None,
        ), action)
        for key in ("measured_effect", "symptom_recovered", "guardrails_passed", "attribution_strength",
                    "measurements", "primary_signals", "guardrail_signals", "load_signals", "control_analysis",
                    "execution_source", "gap_codes", "causal_limitations", "result"):
            value = recomputed.get(key, {}) if key == "control_analysis" else recomputed[key]
            expected = getattr(result, key)
            if isinstance(value, tuple):
                value = list(value)
            if isinstance(expected, tuple):
                expected = list(expected)
            if value != expected:
                return None
        return result
    except (AIOpsApplicationError, KeyError, TypeError, ValueError, AttributeError):
        return None
