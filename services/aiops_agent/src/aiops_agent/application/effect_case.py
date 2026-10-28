"""由同源调查、权威执行及重放对照效果生产待审核成功案例。"""

from uuid import UUID

from aiops_agent.application.changes.effect import effect_source
from aiops_agent.application.changes.effect_replay import replay_supported_effect
from aiops_agent.application.configuration.common import sha256_json
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.errors import state_conflict
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.contracts.change import ProposalOutcome
from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
from aiops_agent.contracts.investigation import InvestigationRecord
from aiops_agent.contracts.turn_answer import TurnEvidenceFact


async def profile_from_effect_report(*, uow, domain_id, report, content, sources):
    if report.report_type != "COMPARISON" or not content.get("scope", {}).get("comparison_result_artifact_id"):
        return None
    if not report.is_current or report.status != "READY":
        raise state_conflict("效果报告不是当前可发布版本")
    run = await uow.runs.get_run_scoped(ops_run_id=report.ops_run_id, domain_id=domain_id)
    if run is None:
        raise state_conflict("效果报告缺少当前调查来源")
    artifact = await effect_source(
        uow=uow, artifact_id=UUID(content["scope"]["comparison_result_artifact_id"]),
        run=run, schema="COMPARISON_RESULT.v1",
    )
    result = await replay_supported_effect(uow=uow, run=run, artifact=artifact)
    if result is None:
        # 没有对照支持时仍走普通诊断参考入口，不抬高验证等级。
        return None
    refs = {str(row.source_artifact_id): row.content_hash for row in sources}
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
    if not await validate_investigation_sources(uow=uow, run=run, payload=root.payload_json):
        raise state_conflict("调查原始证据复核未通过")
    record = InvestigationRecord.model_validate(root.payload_json.get("investigation_record"))
    if record.run_id != str(run.ops_run_id) or record.target_id != str(run.target_id):
        raise state_conflict("效果案例的调查记录归属不一致")
    record.validate_observations(tuple(TurnEvidenceFact.model_validate(value) for value in root.payload_json.get("evidence", ())))
    snapshot = ProposalOutcome.model_validate(snapshot_artifact.payload_json).proposal
    if snapshot is None:
        raise state_conflict("效果案例缺少实际执行提案")
    receipt = await effect_source(uow=uow, artifact_id=result.source_result_artifact_id, run=run)
    if (str(snapshot.proposal_id) != str(result.proposal_id) or str(snapshot.run_id) != str(run.ops_run_id)
        or str(snapshot.target_id) != str(run.target_id)
        or snapshot.proposal_hash != receipt.payload_json["proposal_hash"]
        or snapshot.command_hash != receipt.payload_json["command_hash"]):
        raise state_conflict("效果案例的动作与实际执行不一致")
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
    before_refs = [str(frozen_plan["baseline_artifact_id"])]
    before_ids = set(before_refs)
    controls = set(frozen_plan["control_source_hashes"])
    after_refs = [ref["artifact_id"] for ref in result.measurement_artifact_refs
                  if ref["schema_version"] == "AIOPS_WORKLOAD_REPORT.v2" and ref["artifact_id"] not in before_ids | controls]
    return DiagnosisCaseProfile(
        case_kind="VERIFIED_RESOLUTION" if result.symptom_recovered is True else "VERIFIED_OPTIMIZATION",
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
                  "evidence_refs": [str(snapshot_artifact.artifact_id), str(receipt.artifact_id)]},),
        verification={
            "controlled_verification_ref": f"artifact:{artifact.artifact_id}", "comparability": "COMPARABLE",
            "method": result.control_analysis["method"], "attribution_strength": result.attribution_strength,
            "measured_effect": result.measured_effect, "result": result.result,
            "symptom_recovered": result.symptom_recovered, "guardrails_passed": result.guardrails_passed,
            "before_refs": before_refs, "after_refs": after_refs,
            "guardrail_refs": [str(artifact.artifact_id)], "causal_limitations": list(result.causal_limitations),
        },
        applicability=("仅适用于与冻结验证窗口的业务负载、对象和版本可比的环境",),
        contraindications=result.causal_limitations,
        source={"report_id": str(report.report_id), "run_ids": sorted({str(row.ops_run_id) for row in sources}),
                "content_hash": report.content_hash,
                "investigation_record_id": normalized["investigation_record"].get("record_id")},
    )
