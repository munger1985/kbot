"""由同源调查与重放效果生产四类待审核案例，不把改善或退化自动当作因果。"""

from uuid import UUID

from aiops_agent.application.changes.effect import effect_source
from aiops_agent.application.changes.effect_replay import is_supported_effect, replay_effect
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.errors import state_conflict
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.contracts.change import ProposalOutcome
from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
from aiops_agent.contracts.investigation import InvestigationRecord
from aiops_agent.contracts.turn_answer import TurnEvidenceFact


def effect_case_kind(result):
    if is_supported_effect(result):
        return "VERIFIED_RESOLUTION" if result.symptom_recovered is True else "VERIFIED_OPTIMIZATION"
    if result.direct_effect in {"NOT_ACHIEVED", "ADVERSE"} or result.result in {"UNCHANGED", "DEGRADED"}:
        return "NEGATIVE_EXPERIENCE"
    return "DIAGNOSTIC_REFERENCE"


def effect_case_verification(*, result, artifact_id, frozen_plan):
    """保留每个独立维度和限定，不将未达到阈值写成措施完全无效。"""
    before_refs = [str(frozen_plan["baseline_artifact_id"])]
    controls = set(frozen_plan.get("control_source_hashes") or {})
    after_refs = [ref["artifact_id"] for ref in result.measurement_artifact_refs
        if ref["schema_version"] == "AIOPS_WORKLOAD_REPORT.v2" and ref["artifact_id"] not in set(before_refs) | controls]
    verification = {
        "effect_result_ref": f"artifact:{artifact_id}",
        "comparability": "UNKNOWN" if result.gap_codes else "COMPARABLE",
        "method": result.control_analysis.get("method", "OBSERVATIONAL"),
        "attribution_strength": result.attribution_strength,
        "direct_effect": result.direct_effect, "measured_effect": result.measured_effect,
        "result": result.direct_effect if result.direct_effect in {"NOT_ACHIEVED", "ADVERSE"} else result.result,
        "comparison_result": result.result, "execution_source": result.execution_source,
        "symptom_recovered": result.symptom_recovered, "guardrails_passed": result.guardrails_passed,
        "primary_signals": result.primary_signals, "guardrail_signals": result.guardrail_signals,
        "load_signals": result.load_signals, "gap_codes": list(result.gap_codes),
        "measurements": {name: {key: value[key] for key in (
            "role", "metric", "baseline", "after", "relative_change", "unit", "passed",
        ) if key in value} for name, value in result.measurements.items()},
        "before_refs": before_refs, "after_refs": after_refs, "guardrail_refs": [str(artifact_id)],
        "causal_limitations": list(result.causal_limitations),
    }
    if is_supported_effect(result):
        verification["controlled_verification_ref"] = f"artifact:{artifact_id}"
    return verification


async def profile_from_effect_report(*, uow, domain_id, report, content, sources, plan_capture_reader=None):
    if report.report_type != "COMPARISON" or not content.get("scope", {}).get("comparison_result_artifact_id"):
        return None
    if not report.is_current or report.status not in {"READY", "PARTIAL"}:
        raise state_conflict("效果报告不是当前可发布版本")
    run = await uow.runs.get_run_scoped(ops_run_id=report.ops_run_id, domain_id=domain_id)
    if run is None:
        raise state_conflict("效果报告缺少当前调查来源")
    artifact = await effect_source(
        uow=uow, artifact_id=UUID(content["scope"]["comparison_result_artifact_id"]),
        run=run, schema="COMPARISON_RESULT.v1",
    )
    result = await replay_effect(uow=uow, run=run, artifact=artifact)
    if result is None:
        raise state_conflict("效果原始来源或冻结测量复算未通过")
    refs = {str(row.source_artifact_id): row.content_hash for row in sources}
    if any(refs.get(ref["artifact_id"]) != ref["content_hash"] for ref in result.measurement_artifact_refs):
        raise state_conflict("效果案例缺少冻结报告登记的原始测量来源")
    proposal = await uow.changes.get_proposal(proposal_id=result.proposal_id)
    root_id = getattr(run, "final_artifact_id", None)
    snapshot_id = getattr(proposal, "snapshot_artifact_id", None)
    if root_id is None or snapshot_id is None or str(artifact.artifact_id) not in refs:
        raise state_conflict("效果案例缺少报告关联的调查或执行提案")
    root = await effect_source(uow=uow, artifact_id=root_id, run=run, schema="AIOPS_TURN_RESULT.v1")
    snapshot_artifact = await effect_source(uow=uow, artifact_id=snapshot_id, run=run, schema="PROPOSAL_OUTCOME.v1")
    for source in (artifact, root, snapshot_artifact):
        if refs.get(str(source.artifact_id)) != source.content_hash or sha256_json(source.payload_json) != source.content_hash:
            raise state_conflict("效果案例的原始来源未纳入冻结报告")
    if not await validate_investigation_sources(uow=uow, run=run, payload=root.payload_json, plan_capture_reader=plan_capture_reader):
        raise state_conflict("调查原始证据复核未通过")
    record = InvestigationRecord.model_validate(root.payload_json.get("investigation_record"))
    if record.run_id != str(run.ops_run_id) or record.target_id != str(run.target_id):
        raise state_conflict("效果案例的调查记录归属不一致")
    record.validate_observations(tuple(TurnEvidenceFact.model_validate(value) for value in root.payload_json.get("evidence", ())))
    snapshot = ProposalOutcome.model_validate(snapshot_artifact.payload_json).proposal
    if snapshot is None:
        raise state_conflict("效果案例缺少实际执行提案")
    # 人工回执已由重放验证；不能用只接受 SOURCE_VERIFIED 的读取器将其冒充受控执行。
    receipt = await uow.runs.get_artifact(artifact_id=result.source_result_artifact_id)
    if receipt is None:
        raise state_conflict("效果案例的动作回执已不可读取")
    if (str(snapshot.proposal_id) != str(result.proposal_id) or str(snapshot.run_id) != str(run.ops_run_id)
        or str(snapshot.target_id) != str(run.target_id)):
        raise state_conflict("效果案例的动作与实际执行不一致")
    if result.execution_source == "SOURCE_VERIFIED" and (
        snapshot.proposal_hash != receipt.payload_json["proposal_hash"]
        or snapshot.command_hash != receipt.payload_json["command_hash"]
    ):
        raise state_conflict("效果案例的受控命令与权威执行不一致")
    normalized = normalize_report_source(schema_version=root.schema_version, payload=root.payload_json, source_kind="CHAT")
    if not normalized.get("investigation_record") or not any(
        value.get("level") in {"OBSERVED", "SUPPORTED"} and value.get("evidence_refs")
        for value in normalized.get("facts", ())
    ):
        raise state_conflict("没有可追溯调查判断，不能由效果报告伪造根因")
    target = await uow.targets.get_scoped(target_id=run.target_id, domain_id=domain_id)
    if target is None:
        raise state_conflict("效果案例目标不可见")
    frozen_plan = (await effect_source(uow=uow, artifact_id=result.comparison_plan_artifact_id, run=run)).payload_json["business_effect"]
    return DiagnosisCaseProfile(
        case_kind=effect_case_kind(result),
        problem_signature={
            "problem_class": normalized.get("problem_class") or "DATABASE_OPERATIONS",
            "normalized_symptoms": [value["summary"] for value in normalized.get("facts", ()) if value.get("summary")] or [report.title],
            "signal_names": list(result.primary_signals),
        },
        environment={"database_type": target.db_type, "database_version": target.version_code,
                     "platform": target.environment, "topology": (target.capabilities_json or {}).get("topology")},
        root_cause={
            "summary": normalized["diagnosis_rationale"], "confidence": "UNKNOWN",
            "effective_level": dict(normalized.get("root_cause") or {}).get("effective_level", "INCONCLUSIVE"),
            "evidence_refs": [str(root.artifact_id)],
        },
        actions=({"summary": snapshot.impact or snapshot.rationale,
                  "action_template_id": snapshot.action_template_id,
                  "source_kind": result.execution_source,
                  "evidence_refs": [str(snapshot_artifact.artifact_id), str(receipt.artifact_id)]},),
        verification=effect_case_verification(result=result, artifact_id=artifact.artifact_id, frozen_plan=frozen_plan),
        applicability=("仅适用于与冻结验证窗口的业务负载、对象和版本可比的环境",),
        contraindications=result.causal_limitations,
        source={"report_id": str(report.report_id), "run_ids": sorted({str(row.ops_run_id) for row in sources}),
                "content_hash": report.content_hash,
                "effect_result_artifact_id": str(artifact.artifact_id),
                "investigation_record_id": normalized["investigation_record"].get("record_id")},
    )
